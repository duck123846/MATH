LeTID 兩步一階動力學模型（工程數學期中考）

用一階常微分方程組描述 p 型 PERC 太陽電池的 LeTID（光致與高溫致劣化）：

da/dt = −k₁a
db/dt = k₁a − k₂b
c = 1 − a − b

解析解：b(t) = k₁/(k₂−k₁)·(e^(−k₁t) − e^(−k₂t))；最大劣化時間 t* = ln(k₂/k₁)/(k₂−k₁)。

執行
pip install numpy matplotlib
python letid_full.py

會印出各組參數的 t*、b_max、b(504 h)、t_θ，做數值驗證，並存出 letid_b_of_t.png。

圖中文字需要中文字型

Windows、macOS 內建中文字型，可直接執行。Linux（例如 GitHub Codespaces）請先安裝，否則圖上的中文會顯示成方框：

sudo apt-get update && sudo apt-get install -y fonts-noto-cjk
rm -rf ~/.cache/matplotlib
資料來源
速率式與參數：Repins 等人，MRS Bulletin 48, 589（2023）式 4–6、圖 1。
載子壽命、厚度、複合速度的範例值：NREL PVDeg。這些不是被評估電池的量測值。
暗注入電流：TSEC TSSCN（M10）電池規格書（2018 Preliminary）。
測試條件：IEC TS 63342（標準本文未讀，經 Karas & Repins, 2022 轉述）。
對照參數組：Bredemeier 等人（2017）；程式中的「Kwapil 2017」條目與主模型數字相同，不是獨立結果。
限制

速率常數由高溫外推到 75°C；結果以 B 態比例表示，未換算成功率損失。# -
