from keyhole import scoring as S


def test_norm_and_match():
    assert S.is_correct(" Kabul", ["Kabul"])
    assert S.is_correct(" kabul.", ["Kabul"])
    assert S.is_correct(" Ulan Bator", ["Ulaanbaatar", "Ulan Bator"])
    assert S.is_correct(" Brasília", ["Brasilia"])
    assert S.is_correct(" der Hund", ["Hund"])
    assert S.is_correct(" 1,234", ["1234"])
    assert not S.is_correct(" The capital is Kabul", ["Kabul"])
    assert S.is_correct(" Kabul is the capital", ["Kabul"])
    assert S.is_correct(" Kabul\nmore", ["Kabul"])


def test_answer_present_and_classes():
    assert S.answer_present("The capital city of Iraq is Baghdad.", ["Baghdad"])
    assert not S.answer_present("The logs show repeated timeouts", ["Baghdad"])
    assert S.failure_class("The capital city of Iraq is Baghdad.", ["Baghdad"], "FACT", "") == "F0"
    assert S.failure_class("I cannot help with that", ["Baghdad"], "FACT", "") == "F0"
    assert S.failure_class("Atlanta", ["Tbilisi"], "FACT", "") == "F2"
    assert S.failure_class("{\"a\": 1}", ["Tbilisi"], "FACT", "") == "F0"
    assert S.failure_class("forty", ["40"], "ARITH", "") == "F0"
    assert S.failure_class("Mateo Castillo", ["Paris"], "FACT", "from: mateo castillo wrote") == "F1"
