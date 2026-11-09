"""RAG 搜尋測試：問題有沒有找到正確的規章段落（只呼叫 embedding 模型，不呼叫聊天模型）。

執行方式：  python test_rag.py
- 規章有寫的問題：預期的段落必須在交給模型的前 TOP_K 段裡（分數 >= MIN_SCORE）
  （「入住前颱風」和「住宿期間颱風」用詞很像、分數常常幾乎平手，所以一次給模型 2 段當緩衝）
- 規章沒寫的問題：最高分必須 < MIN_SCORE（found = False），模型才不會拿不相關的段落硬湊答案
印出每題的分數，用來決定 app.py 裡的 MIN_SCORE。
"""

import sys

import numpy as np

from app import MIN_SCORE, RULE_CHUNKS, SEARCH_KEYS, embed, rule_scores, search_bnb_rules

sys.stdout.reconfigure(encoding="utf-8")

# (問題, 預期第 1 名的段落標題開頭；None = 規章沒寫，應該找不到)
TEST_SET = [
    ("入住當天有颱風，訂金可以退嗎？",                        "Typhoon before arrival"),
    ("A typhoon is coming on my check-in day. Can I get my deposit back?", "Typhoon before arrival"),
    ("只有海上颱風警報，可以改期嗎？",                        "Typhoon before arrival"),
    ("住到一半颱風來了走不了，多住的晚上怎麼算？",              "Typhoon during your stay"),
    ("颱風天浮潛取消了會退錢嗎？",                            "Typhoon during your stay"),
    ("入住押金要付多少？什麼時候退？",                        "Security deposit"),
    ("How much is the security deposit and when do I get it back?", "Security deposit"),
    ("我打破杯子又弄丟房卡，要賠多少？",                      "Damage and compensation"),
    ("I spilled coffee on the bed sheet and it will not wash out. What is the charge?", "Damage and compensation"),
    ("在房間抽電子菸會怎樣？",                                "Smoking and special cleaning"),
    ("到了現場卻說沒有房間怎麼辦？",                          "Overbooking"),     # 兩段超額預訂都算對
    ("My room's air conditioner is broken. Will you give me another room?", "Overbooking or a room problem"),
    ("如果你們安排我去別家飯店，計程車錢誰付？",                "Overbooking: moving"),
    ("提前 10 天取消可以退多少訂金？",                        "Booking deposit"),
    # 規章沒寫到的
    ("可以在民宿辦婚禮派對嗎？",                              None),
    ("What will the weather be like in Kenting next week?",   None),
    ("推薦我墾丁好吃的海鮮餐廳",                              None),
    ("Do you have a swimming pool?",                          None),
]

# 檢查 0：每把搜尋鑰匙都沒有被 embedding 模型截斷（尾端加上無關文字，向量應該要改變）
JUNK = "\nZZZ unrelated text about pizza and football." * 5
for key in SEARCH_KEYS:
    a, b = embed([key, key + JUNK])
    same = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    if same > 0.9999:
        print(f"WARNING: this search key is too long and gets cut ({same:.5f}): {key[:40]!r}")
print(f"Checked {len(SEARCH_KEYS)} search keys for truncation.\n")

correct = 0
for question, expected in TEST_SET:
    scores = rule_scores(question)
    order = np.argsort(scores)[::-1]
    top_title, top_score = RULE_CHUNKS[order[0]]["title"], float(scores[order[0]])
    result = search_bnb_rules(question)
    given = [c["title"] for c in result.get("chunks", [])]           # 實際交給模型的段落
    if expected is None:
        ok = not result["found"]
    else:
        ok = any(title.startswith(expected) for title in given)
    correct += ok
    print(f"{'OK  ' if ok else 'MISS'} {top_score:.3f}  2nd {float(scores[order[1]]):.3f}  {question}")
    print(f"      top: {top_title}   expected: {expected or '(not in the rules)'}")

print(f"\nMIN_SCORE = {MIN_SCORE}")
print(f"RAG retrieval = {correct}/{len(TEST_SET)} = {correct / len(TEST_SET):.0%}")
