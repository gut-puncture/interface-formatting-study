from __future__ import annotations

import csv
import difflib
import hashlib
import io
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass(frozen=True)
class OptionSlot:
    content_id: int
    label_spans: tuple[SourceSpan, ...]
    payload_spans: tuple[SourceSpan, ...]

    def payload_texts(self, source: str) -> tuple[str, ...]:
        return tuple(span.text(source) for span in self.payload_spans)


@dataclass(frozen=True)
class OptionRepresentation:
    slots: tuple[OptionSlot, OptionSlot, OptionSlot, OptionSlot]


@dataclass(frozen=True)
class ParsedPrompt:
    wrapper_name: str
    source_sha256: str
    representations: tuple[OptionRepresentation, ...]
    separable: bool
    not_applicable_reason: str
    provenance: str = "deterministic"


def _span(match: re.Match[str], group: str | int = 0) -> SourceSpan:
    start, end = match.span(group)
    return SourceSpan(start, end)


def _validate_span(source: str, span: SourceSpan) -> None:
    if span.start < 0 or span.end <= span.start or span.end > len(source):
        raise ValueError(f"invalid source span: {span}")


def validate_option_map(parsed: ParsedPrompt, source: str) -> None:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if parsed.source_sha256 != digest:
        raise ValueError("source prompt checksum mismatch")
    if not parsed.representations and not parsed.separable:
        return
    if not parsed.representations:
        raise ValueError("option map has no representations")
    for representation in parsed.representations:
        if len(representation.slots) != 4:
            raise ValueError("option representation must contain four slots")
        content_ids = [slot.content_id for slot in representation.slots]
        if sorted(content_ids) != [0, 1, 2, 3]:
            raise ValueError("option representation content IDs are not a permutation")
        spans: list[SourceSpan] = []
        for position, slot in enumerate(representation.slots):
            if parsed.separable and not slot.label_spans:
                raise ValueError("separable option slot has no label span")
            expected_label = LETTERS[position]
            if any(span.text(source).upper() != expected_label for span in slot.label_spans):
                raise ValueError(f"option label span does not contain {expected_label}")
            for candidate in (*slot.label_spans, *slot.payload_spans):
                _validate_span(source, candidate)
                spans.append(candidate)
        ordered = sorted(spans, key=lambda candidate: (candidate.start, candidate.end))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start < previous.end:
                raise ValueError("option-map spans overlap")


def _make_parsed(
    source: str,
    wrapper_name: str,
    representations: Iterable[OptionRepresentation],
    *,
    separable: bool,
    reason: str = "",
) -> ParsedPrompt:
    parsed = ParsedPrompt(
        wrapper_name=wrapper_name,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        representations=tuple(representations),
        separable=separable,
        not_applicable_reason=reason,
    )
    validate_option_map(parsed, source)
    return parsed


def _line_bounds(source: str) -> Iterable[tuple[int, int, str]]:
    cursor = 0
    for line in source.splitlines(keepends=True):
        end = cursor + len(line)
        yield cursor, end, line.rstrip("\r\n")
        cursor = end
    if cursor < len(source):
        yield cursor, len(source), source[cursor:]


def _field_spans(line: str, line_start: int) -> list[tuple[str, SourceSpan]]:
    reader = csv.reader(io.StringIO(line))
    fields = next(reader)
    spans: list[tuple[str, SourceSpan]] = []
    cursor = 0
    for field in fields:
        candidates = (field, f'"{field.replace(chr(34), chr(34) * 2)}"')
        found = None
        for candidate in candidates:
            index = line.find(candidate, cursor)
            if index >= 0:
                found = (index, index + len(candidate), candidate)
                break
        if found is None:
            raise ValueError("csv_field_span_not_found")
        start, end, raw = found
        payload_start = start + (1 if raw.startswith('"') else 0)
        payload_end = end - (1 if raw.endswith('"') else 0)
        spans.append((field, SourceSpan(line_start + payload_start, line_start + payload_end)))
        cursor = end + (1 if end < len(line) and line[end : end + 1] == "," else 0)
    return spans


def _parse_csv(source: str, choices: Sequence[str]) -> ParsedPrompt:
    candidates: dict[str, list[tuple[SourceSpan, SourceSpan]]] = {label: [] for label in LETTERS}
    for start, _end, line in _line_bounds(source):
        if not line or "," not in line:
            continue
        try:
            fields = _field_spans(line, start)
        except (csv.Error, ValueError):
            fields = []
        label_fields = [(value.strip().upper(), span) for value, span in fields if value.strip().upper() in LETTERS]
        if len(label_fields) == 1:
            label, label_span = label_fields[0]
            choice = str(choices[LETTERS.index(label)])
            exact = [(value, span) for value, span in fields if value == choice and span != label_span]
            if len(exact) == 1:
                candidates[label].append((label_span, exact[0][1]))

        # Some source prompts intentionally use human-readable, unquoted CSV.
        # In those rows commas inside an option are payload bytes, not delimiters.
        edge_label = re.match(r"^\s*(?P<label>[A-D])\s*,", line)
        if edge_label is None:
            edge_label = re.search(r",\s*(?P<label>[A-D])\s*$", line)
        if edge_label is None:
            continue
        label = edge_label.group("label")
        choice = str(choices[LETTERS.index(label)])
        occurrences = list(re.finditer(re.escape(choice), line))
        if len(occurrences) == 1:
            choice_match = occurrences[0]
            candidates[label].append(
                (
                    SourceSpan(start + edge_label.start("label"), start + edge_label.end("label")),
                    SourceSpan(start + choice_match.start(), start + choice_match.end()),
                )
            )
    records: dict[str, tuple[SourceSpan, SourceSpan]] = {}
    for label, found in candidates.items():
        unique = list(dict.fromkeys(found))
        if len(unique) != 1:
            raise ValueError("option_structure_not_resolved")
        records[label] = unique[0]
    if set(records) != set(LETTERS):
        raise ValueError("option_structure_not_resolved")
    slots = tuple(
        OptionSlot(index, (records[label][0],), (records[label][1],))
        for index, label in enumerate(LETTERS)
    )
    return _make_parsed(source, "csv_inline", (OptionRepresentation(slots),), separable=True)


def _quoted_array_values(source: str) -> tuple[SourceSpan, ...]:
    array = re.search(r"(?is)\b(?:options|choices)\s*:\s*\[(?P<body>.*?)\]", source)
    if array is None:
        return ()
    values = [
        SourceSpan(match.start("value") + array.start("body"), match.end("value") + array.start("body"))
        for match in re.finditer(r'"(?P<value>(?:\\.|[^"\\])*)"', array.group("body"))
    ]
    return tuple(values) if len(values) == 4 else ()


def _parse_graphql(source: str, choices: Sequence[str]) -> ParsedPrompt:
    values = _quoted_array_values(source)
    if values:
        slots = tuple(OptionSlot(index, (), (values[index],)) for index in range(4))
        return _make_parsed(
            source,
            "graphql_query",
            (OptionRepresentation(slots),),
            separable=False,
            reason="independent_label_position_not_identifiable",
        )
    pattern = re.compile(
        r'(?is)(?:letter|id|choice|option)\s*:\s*"?(?P<label>[A-D])"?'
        r'.{0,80}?(?:value|text|description)\s*:\s*"(?P<payload>(?:\\.|[^"\\])*)"'
    )
    matches = list(pattern.finditer(source))
    if {match.group("label") for match in matches} != set(LETTERS):
        object_region = re.search(r"(?is)\boptions\s*:\s*\{(?P<body>.*?)\}", source)
        if object_region is not None:
            matches = list(
                re.finditer(
                    r'(?is)(?<![A-Za-z0-9_])(?P<label>[A-D])\s*:\s*"(?P<payload>(?:\\.|[^"\\])*)"',
                    object_region.group("body"),
                )
            )
            offset = object_region.start("body")
            if {match.group("label") for match in matches} == set(LETTERS):
                by_label = {match.group("label"): match for match in matches}
                slots = tuple(
                    OptionSlot(
                        index,
                        (SourceSpan(offset + by_label[label].start("label"), offset + by_label[label].end("label")),),
                        (SourceSpan(offset + by_label[label].start("payload"), offset + by_label[label].end("payload")),),
                    )
                    for index, label in enumerate(LETTERS)
                )
                return _make_parsed(
                    source,
                    "graphql_query",
                    (OptionRepresentation(slots),),
                    separable=True,
                )
    if {match.group("label") for match in matches} != set(LETTERS):
        raise ValueError("option_structure_not_resolved")
    by_label = {match.group("label"): match for match in matches}
    slots = tuple(
        OptionSlot(index, (_span(by_label[label], "label"),), (_span(by_label[label], "payload"),))
        for index, label in enumerate(LETTERS)
    )
    return _make_parsed(source, "graphql_query", (OptionRepresentation(slots),), separable=True)


def _parse_html(source: str, choices: Sequence[str]) -> ParsedPrompt:
    containers = list(re.finditer(r"(?is)<(?P<tag>option|tr)\b[^>]*>.*?</(?P=tag)>", source))
    option_slots: list[OptionSlot] = []
    for index, label in enumerate(LETTERS):
        candidates: list[tuple[re.Match[str], re.Match[str], re.Match[str], SourceSpan]] = []
        for container in containers:
            if container.group("tag").lower() != "option":
                continue
            raw = container.group(0)
            opening_end = raw.find(">")
            closing_start = raw.lower().rfind("</option>")
            value_label = re.search(
                rf'(?i)\bvalue\s*=\s*["\'](?P<label>{label})["\']', raw[: opening_end + 1]
            )
            visible_label = re.match(
                rf"(?is)\s*(?P<label>{label})\s*[\):.-]\s*",
                raw[opening_end + 1 : closing_start],
            )
            if value_label is None or visible_label is None:
                continue
            payload_start = opening_end + 1 + visible_label.end()
            payload_end = closing_start
            while payload_end > payload_start and raw[payload_end - 1].isspace():
                payload_end -= 1
            candidates.append(
                (
                    container,
                    value_label,
                    visible_label,
                    SourceSpan(container.start() + payload_start, container.start() + payload_end),
                )
            )
        if len(candidates) != 1:
            option_slots = []
            break
        container, value_label, visible_label, payload_span = candidates[0]
        opening_end = container.group(0).find(">")
        option_slots.append(
            OptionSlot(
                index,
                (
                    SourceSpan(
                        container.start() + value_label.start("label"),
                        container.start() + value_label.end("label"),
                    ),
                    SourceSpan(
                        container.start() + opening_end + 1 + visible_label.start("label"),
                        container.start() + opening_end + 1 + visible_label.end("label"),
                    ),
                ),
                (payload_span,),
            )
        )
    if len(option_slots) == 4:
        return _make_parsed(
            source,
            "html_form",
            (OptionRepresentation(tuple(option_slots)),),
            separable=True,
        )

    slots: list[OptionSlot] = []
    for index, label in enumerate(LETTERS):
        candidates = []
        for container in containers:
            labels = [
                _span(match, "label")
                for match in re.finditer(
                    rf'(?i)(?:value\s*=\s*["\']|(?<![A-Za-z0-9_]))(?P<label>{label})(?=["\']|\s*[\):])',
                    container.group(0),
                )
            ]
            labels = tuple(
                SourceSpan(span.start + container.start(), span.end + container.start()) for span in labels
            )
            if labels:
                candidates.append((container, labels))
        if len(candidates) != 1:
            raise ValueError("option_structure_not_resolved")
        container, labels = candidates[0]
        payloads: list[SourceSpan] = []
        for match in re.finditer(r"(?is)>(?P<text>[^<>]+)<", container.group(0)):
            raw_text = match.group("text")
            text = raw_text.strip()
            if not text or re.fullmatch(rf"{label}\)?", text):
                continue
            prefix = re.match(rf"(?i)\s*{label}\s*[\):.-]?\s*", raw_text)
            prefix_end = prefix.end() if prefix is not None else len(raw_text) - len(raw_text.lstrip())
            text = raw_text[prefix_end:].strip()
            if not text:
                continue
            offset = prefix_end + len(raw_text[prefix_end:]) - len(raw_text[prefix_end:].lstrip())
            start = container.start() + match.start("text") + offset
            payloads.append(SourceSpan(start, start + len(text)))
        slots.append(OptionSlot(index, labels, tuple(payloads)))
    return _make_parsed(source, "html_form", (OptionRepresentation(tuple(slots)),), separable=True)


def _labelled_line_records(source: str, wrapper_name: str) -> list[tuple[str, SourceSpan, SourceSpan | None]]:
    patterns = {
        "ini_file": r"(?i)^(?:\[\s*(?:option[_ .-]*)?(?P<label>[A-D])\s*\]|(?:option[_ .-]*)?(?P<label2>[A-D])\s*=)(?P<payload>.*)$",
        "key_equals": r"(?i)^(?:OPTION_)?(?P<label>[A-D])\s*=\s*(?P<payload>.*)$",
        "shell_heredoc": r'''(?i)^(?:OPTION_)?(?P<label>[A-D])(?:\)|\s*=)\s*["']?(?P<payload>.*?)["']?$''',
        "toml_config": r'''(?i)^(?:option[_ .-]*)?["']?(?P<label>[A-D])["']?\s*=\s*["']?(?P<payload>.*?)["']?$''',
        "protobuf_msg": r'''(?i)^.*?\b(?:OPTION[_ ]?|option)(?P<label>[A-D])\b(?P<middle>.*?)(?::|=)\s*["']?(?P<payload>.*?)["']?[,;]?$''',
    }
    regex = re.compile(patterns[wrapper_name])
    protobuf_string = re.compile(
        r'''(?i)^.*?\(option\)\s*=\s*["'](?P<label>[A-D])\)\s*(?P<payload>.*?)["']\s*[\],;]*\s*$'''
    )
    records = []
    for start, _end, line in _line_bounds(source):
        match = protobuf_string.match(line) if wrapper_name == "protobuf_msg" else None
        if match is not None:
            records.append(
                (
                    match.group("label").upper(),
                    SourceSpan(start + match.start("label"), start + match.end("label")),
                    SourceSpan(start + match.start("payload"), start + match.end("payload")),
                )
            )
            continue
        stripped = line.strip()
        match = regex.match(stripped)
        if match is None:
            continue
        label_group = "label" if match.groupdict().get("label") else "label2"
        label = match.group(label_group).upper()
        left_trim = len(line) - len(line.lstrip())
        label_span = SourceSpan(start + left_trim + match.start(label_group), start + left_trim + match.end(label_group))
        payload = match.groupdict().get("payload")
        payload_span = None
        if payload is not None and payload.strip():
            payload_offset = len(payload) - len(payload.lstrip())
            payload_text = payload.strip().rstrip("\"',;")
            payload_start = start + left_trim + match.start("payload") + payload_offset
            payload_span = SourceSpan(payload_start, payload_start + len(payload_text))
        records.append((label, label_span, payload_span))
    return records


def _representations_from_records(
    records: Sequence[tuple[str, SourceSpan, SourceSpan | None]],
) -> tuple[OptionRepresentation, ...]:
    representations: list[OptionRepresentation] = []
    current: dict[str, tuple[SourceSpan, SourceSpan | None]] = {}
    for label, label_span, payload_span in records:
        if label in current:
            if set(current) == set(LETTERS):
                slots = tuple(
                    OptionSlot(index, (current[key][0],), (() if current[key][1] is None else (current[key][1],)))
                    for index, key in enumerate(LETTERS)
                )
                representations.append(OptionRepresentation(slots))
                current = {}
            else:
                current = {}
        current[label] = (label_span, payload_span)
    if set(current) == set(LETTERS):
        slots = tuple(
            OptionSlot(index, (current[key][0],), (() if current[key][1] is None else (current[key][1],)))
            for index, key in enumerate(LETTERS)
        )
        representations.append(OptionRepresentation(slots))
    return tuple(representations)


def _parse_labelled_lines(source: str, wrapper_name: str) -> ParsedPrompt:
    representations = _representations_from_records(_labelled_line_records(source, wrapper_name))
    if not representations:
        raise ValueError("option_structure_not_resolved")
    return _make_parsed(source, wrapper_name, representations, separable=True)


def _parse_ini(source: str) -> ParsedPrompt:
    headers = list(
        re.finditer(r"(?im)^\s*\[\s*option[_ .-]*(?P<label>[A-D])\s*\]\s*$", source)
    )
    if {header.group("label").upper() for header in headers} == set(LETTERS):
        slots: list[OptionSlot] = []
        by_label = {header.group("label").upper(): header for header in headers}
        ordered_headers = sorted(headers, key=lambda header: header.start())
        for index, label in enumerate(LETTERS):
            header = by_label[label]
            header_index = ordered_headers.index(header)
            region_end = (
                ordered_headers[header_index + 1].start()
                if header_index + 1 < len(ordered_headers)
                else len(source)
            )
            body = source[header.end() : region_end]
            value = re.search(r"(?im)^\s*description\s*=\s*(?P<payload>\S.*)\s*$", body)
            if value is None:
                break
            payload_start = header.end() + value.start("payload")
            payload_end = header.end() + value.end("payload")
            slots.append(
                OptionSlot(
                    index,
                    (_span(header, "label"),),
                    (SourceSpan(payload_start, payload_end),),
                )
            )
        if len(slots) == 4:
            return _make_parsed(
                source,
                "ini_file",
                (OptionRepresentation(tuple(slots)),),
                separable=True,
            )
    return _parse_labelled_lines(source, "ini_file")


def _parse_redacted_placeholders(source: str, wrapper_name: str) -> ParsedPrompt:
    payloads: list[SourceSpan] = []
    for label in LETTERS:
        token = f"OPTION_{label}_PLACEHOLDER"
        matches = list(re.finditer(re.escape(token), source))
        if len(matches) != 1:
            raise ValueError("redacted_placeholder_not_unique")
        payloads.append(_span(matches[0]))
    slots: list[OptionSlot] = []
    previous_end = 0
    for index, (label, payload) in enumerate(zip(LETTERS, payloads, strict=True)):
        prefix_start = max(previous_end, payload.start - 120)
        prefix = source[prefix_start : payload.start]
        label_spans: set[SourceSpan] = set()
        for pattern in (
            rf'''(?i)\bvalue\s*=\s*["'](?P<label>{label})["']''',
            rf"(?i)(?<![A-Za-z0-9_])(?P<label>{label})\s*[:)=]",
        ):
            for match in re.finditer(pattern, prefix):
                label_spans.add(
                    SourceSpan(
                        prefix_start + match.start("label"),
                        prefix_start + match.end("label"),
                    )
                )
        if not label_spans:
            raise ValueError("redacted_placeholder_label_not_resolved")
        slots.append(
            OptionSlot(
                index,
                tuple(sorted(label_spans, key=lambda span: span.start)),
                (payload,),
            )
        )
        previous_end = payload.end
    return _make_parsed(
        source,
        wrapper_name,
        (OptionRepresentation(tuple(slots)),),
        separable=True,
    )


def parse_prompt_options(
    source: str,
    wrapper_name: str,
    canonical_choices: Sequence[str],
) -> ParsedPrompt:
    if len(canonical_choices) != 4:
        raise ValueError("option parsing requires four canonical choices")
    if tuple(map(str, canonical_choices)) == tuple(
        f"OPTION_{label}_PLACEHOLDER" for label in LETTERS
    ):
        try:
            return _parse_redacted_placeholders(source, wrapper_name)
        except ValueError:
            pass
    if wrapper_name == "csv_inline":
        return _parse_csv(source, canonical_choices)
    if wrapper_name == "graphql_query":
        return _parse_graphql(source, canonical_choices)
    if wrapper_name == "html_form":
        return _parse_html(source, canonical_choices)
    if wrapper_name == "ini_file":
        return _parse_ini(source)
    if wrapper_name in {"key_equals", "protobuf_msg", "shell_heredoc", "toml_config"}:
        return _parse_labelled_lines(source, wrapper_name)
    if wrapper_name == "plain":
        return _parse_labelled_lines(source, "shell_heredoc")
    raise ValueError(f"unknown_source_wrapper:{wrapper_name}")


def _replace_spans(source: str, replacements: Sequence[tuple[SourceSpan, str]]) -> str:
    ordered = sorted(replacements, key=lambda pair: pair[0].start)
    pieces: list[str] = []
    cursor = 0
    for span, replacement in ordered:
        if span.start < cursor:
            raise ValueError("option-map replacements overlap")
        pieces.extend((source[cursor : span.start], replacement))
        cursor = span.end
    pieces.append(source[cursor:])
    return "".join(pieces)


def transform_with_option_map(
    parsed: ParsedPrompt,
    source: str,
    *,
    position_shift: int,
    label_shift: int,
) -> str:
    validate_option_map(parsed, source)
    if position_shift not in range(4) or label_shift not in range(4):
        raise ValueError("position_shift and label_shift must be in [0, 3]")
    if not parsed.separable and (position_shift or label_shift):
        raise ValueError(parsed.not_applicable_reason)
    replacements: list[tuple[SourceSpan, str]] = []
    for representation in parsed.representations:
        slots = representation.slots
        for position, target in enumerate(slots):
            source_position = (position - position_shift) % 4
            moving = slots[source_position]
            new_label = LETTERS[(moving.content_id + label_shift) % 4]
            replacements.extend((span, new_label) for span in target.label_spans)
            if len(target.payload_spans) != len(moving.payload_spans):
                raise ValueError("option payload representations are inconsistent")
            replacements.extend(
                (target_span, moving_span.text(source))
                for target_span, moving_span in zip(target.payload_spans, moving.payload_spans, strict=True)
            )
    return _replace_spans(source, replacements)


def remap_option_map_by_redaction_diff(
    parsed: ParsedPrompt,
    source: str,
    redacted: str,
) -> ParsedPrompt:
    """Transfer a verified map when redaction changes only payload regions."""

    validate_option_map(parsed, source)
    if not parsed.separable:
        raise ValueError("cannot remap a nonseparable option map")
    opcodes = difflib.SequenceMatcher(a=source, b=redacted, autojunk=False).get_opcodes()

    def unchanged(span: SourceSpan) -> SourceSpan:
        for tag, source_start, source_end, target_start, _target_end in opcodes:
            if tag == "equal" and source_start <= span.start and span.end <= source_end:
                offset = target_start - source_start
                return SourceSpan(span.start + offset, span.end + offset)
        raise ValueError("redaction changed an option label span")

    payload_map: dict[SourceSpan, SourceSpan] = {}
    for content_id, label in enumerate(LETTERS):
        source_spans = sorted(
            (
                span
                for representation in parsed.representations
                for slot in representation.slots
                if slot.content_id == content_id
                for span in slot.payload_spans
            ),
            key=lambda span: span.start,
        )
        token = f"OPTION_{label}_PLACEHOLDER"
        target_spans = [_span(match) for match in re.finditer(re.escape(token), redacted)]
        if len(source_spans) != len(target_spans):
            raise ValueError(f"redaction payload count mismatch for option {label}")
        payload_map.update(zip(source_spans, target_spans, strict=True))

    remapped = ParsedPrompt(
        wrapper_name=parsed.wrapper_name,
        source_sha256=hashlib.sha256(redacted.encode("utf-8")).hexdigest(),
        representations=tuple(
            OptionRepresentation(
                tuple(
                    OptionSlot(
                        slot.content_id,
                        tuple(unchanged(span) for span in slot.label_spans),
                        tuple(payload_map[span] for span in slot.payload_spans),
                    )
                    for slot in representation.slots
                )
            )
            for representation in parsed.representations
        ),
        separable=True,
        not_applicable_reason="",
        provenance=parsed.provenance,
    )
    validate_option_map(remapped, redacted)
    return remapped


def remap_option_map_through_canonical_redaction(
    parsed: ParsedPrompt,
    source: str,
    *,
    question: str,
    choices: Sequence[str],
    redacted: str,
) -> ParsedPrompt:
    """Track intended option spans through the project's exact redaction steps.

    Short answer strings can also occur in wrapper syntax.  The existing
    calibration protocol replaces those occurrences too; event tracking lets
    us preserve that prompt byte-for-byte while mapping only the replacements
    that came from the four verified option payload spans.
    """

    validate_option_map(parsed, source)
    if not parsed.separable:
        raise ValueError("cannot remap a nonseparable option map")
    if len(choices) != 4:
        raise ValueError("canonical redaction requires four choices")

    # Each unit retains the source character it descended from and, for an
    # inserted option placeholder, the unique replacement event that made it.
    units: list[tuple[str, int | None, int | None]] = [
        (character, index, None) for index, character in enumerate(source)
    ]
    event_origins: dict[int, tuple[int, ...]] = {}
    next_event = 0

    def text() -> str:
        return "".join(character for character, _origin, _event in units)

    def replace_matches(pattern: str, replacement: str, *, track_event: bool) -> None:
        nonlocal next_event
        flags = re.IGNORECASE if not track_event else 0
        matches = list(re.finditer(re.escape(pattern), text(), flags=flags))
        for match in reversed(matches):
            removed = units[match.start() : match.end()]
            event = next_event if track_event else None
            if track_event:
                event_origins[next_event] = tuple(
                    origin for _character, origin, _old_event in removed if origin is not None
                )
                next_event += 1
            units[match.start() : match.end()] = [
                (character, None, event) for character in replacement
            ]

    replace_matches(question, "QUESTION_TEXT_PLACEHOLDER", track_event=False)
    for label, choice in zip(LETTERS, choices, strict=True):
        replace_matches(str(choice), f"OPTION_{label}_PLACEHOLDER", track_event=True)
    if text() != redacted:
        raise ValueError("redacted prompt does not match canonical redaction")

    origin_positions: dict[int, list[int]] = {}
    event_positions: dict[int, list[int]] = {}
    for position, (_character, origin, event) in enumerate(units):
        if origin is not None:
            origin_positions.setdefault(origin, []).append(position)
        if event is not None:
            event_positions.setdefault(event, []).append(position)

    def consecutive_span(positions: Sequence[int], *, description: str) -> SourceSpan:
        if not positions or list(positions) != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(f"{description} is not contiguous after redaction")
        return SourceSpan(positions[0], positions[-1] + 1)

    def translate_label(span: SourceSpan) -> SourceSpan:
        positions = [
            position
            for origin in range(span.start, span.end)
            for position in origin_positions.get(origin, ())
        ]
        return consecutive_span(positions, description="option label span")

    payload_events: dict[SourceSpan, int] = {}
    for representation in parsed.representations:
        for slot in representation.slots:
            for span in slot.payload_spans:
                expected = tuple(range(span.start, span.end))
                matches = [event for event, origins in event_origins.items() if origins == expected]
                if len(matches) != 1:
                    raise ValueError("could not identify the intended option replacement event")
                payload_events[span] = matches[0]

    representations = tuple(
        OptionRepresentation(
            tuple(
                OptionSlot(
                    slot.content_id,
                    tuple(translate_label(span) for span in slot.label_spans),
                    tuple(
                        consecutive_span(
                            event_positions[payload_events[span]],
                            description="option payload span",
                        )
                        for span in slot.payload_spans
                    ),
                )
                for slot in representation.slots
            )
        )
        for representation in parsed.representations
    )
    remapped = ParsedPrompt(
        wrapper_name=parsed.wrapper_name,
        source_sha256=hashlib.sha256(redacted.encode("utf-8")).hexdigest(),
        representations=representations,
        separable=True,
        not_applicable_reason="",
        provenance=parsed.provenance,
    )
    validate_option_map(remapped, redacted)
    return remapped
