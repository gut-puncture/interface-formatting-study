from __future__ import annotations

import pandas as pd

from interface_formatting_study.causal_design import ACTIVE_WRAPPERS


_SUFFIX = "\n\nReturn only the letter (A, B, C, or D).\nAnswer: "


def source_prompt_frame() -> pd.DataFrame:
    prompts = {
        "csv_inline": "Option,Text\nA,3\nB,4\nC,5\nD,6",
        "graphql_query": 'query { answer(options: { A: "3", B: "4", C: "5", D: "6" }) }',
        "html_form": (
            '<option value="A">A) 3</option>\n<option value="B">B) 4</option>\n'
            '<option value="C">C) 5</option>\n<option value="D">D) 6</option>'
        ),
        "ini_file": "[options]\nA=3\nB=4\nC=5\nD=6",
        "key_equals": "A=3\nB=4\nC=5\nD=6",
        "protobuf_msg": 'OPTION_A="3"\nOPTION_B="4"\nOPTION_C="5"\nOPTION_D="6"',
        "shell_heredoc": "A) 3\nB) 4\nC) 5\nD) 6",
        "toml_config": '[options]\nA="3"\nB="4"\nC="5"\nD="6"',
    }
    return pd.DataFrame(
        [
            {
                "item_id": "item-1",
                "subject": "math",
                "split": "train",
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "correct_index": 1,
                "wrapper_name": wrapper,
                "wrapped_prompt": prompts[wrapper] + _SUFFIX,
            }
            for wrapper in ACTIVE_WRAPPERS
        ]
    )
