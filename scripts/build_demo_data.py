import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from reviewguard.config import ROOT
from reviewguard.data import load_dataset

TRIO = [
    "Excellent product. Amazing quality and absolutely worth every penny.",
    "Amazing product. Excellent quality and definitely worth every penny.",
    "Excellent quality. Amazing purchase and worth every penny.",
]
EXAMPLES = [
    dict(
        name="Detailed stay",
        text="We stayed in room 412 for three nights. The bed was comfortable and the bathroom had a good shower. Breakfast started at 7 and the train was a ten minute walk. The elevator was slow, but the front desk helped with our bags. I would stay again for the location.",
        rating=4,
    ),
    dict(
        name="Promotional",
        text="BEST PRODUCT EVER!!!!! BUY NOW!!!!! Absolutely perfect, amazing amazing amazing. Must buy, highly recommend, five stars!!!",
        rating=5,
    ),
    dict(name="Template", text=TRIO[0], rating=5),
    dict(
        name="Rating contradiction",
        text="Absolutely wonderful stay. Everything was perfect and the service was amazing. I loved every minute and highly recommend this hotel.",
        rating=1,
    ),
    dict(name="Very short", text="AMAZING!!! PERFECT!!!", rating=5),
    dict(
        name="Long and balanced",
        text="During our four night stay the room on the eighth floor faced the river. The bathroom was clean and breakfast included eggs and fruit. We walked to the station in twelve minutes. Although the bed was comfortable, the air conditioning rattled on the first night. The desk sent a technician the next morning. Overall the location worked well for our conference, but I would ask for a quieter room next time.",
        rating=4,
    ),
]


def main():
    df = load_dataset().sample(180, random_state=42).drop(columns="label")
    df["rating"] = float("nan")
    df["demo_injected"] = False
    df["demo_group"] = ""
    rows = []
    groups = [
        TRIO,
        [
            "This hotel is absolutely perfect. Amazing service, spotless rooms and the best breakfast. Highly recommend, five stars!"
        ]
        * 5,
        [
            "I stayed for two nights. The room was dirty, the shower was broken and the desk ignored my complaint. Never again."
        ]
        * 5,
    ]
    for g, texts in enumerate(groups):
        for text in texts:
            rows.append(
                dict(
                    text=text,
                    rating=5 if g < 2 else 5,
                    deceptive="synthetic",
                    hotel="demo_hotel",
                    polarity="positive" if g < 2 else "negative",
                    source="synthetic_demo",
                    demo_injected=True,
                    demo_group=f"group_{g+1}",
                )
            )
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(ROOT / "data/samples/demo_batch.csv", index=False)
    pd.DataFrame(
        {
            "text": TRIO,
            "rating": [5] * 3,
            "demo_injected": [True] * 3,
            "demo_group": ["motivating_example"] * 3,
        }
    ).to_csv(ROOT / "data/samples/three_review_demo.csv", index=False)
    (ROOT / "data/samples/single_examples.json").write_text(
        json.dumps(EXAMPLES, indent=2)
    )


if __name__ == "__main__":
    main()
