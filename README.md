# 項目簡介 (Project Overview)：
闡述聽障學童的語言剝奪危機，以及本系統如何結合邊緣端 AI 提供零成本的數理與語文遊戲化學習。

技術架構 (Tech Stack)：提及 MediaPipe、Pygame、Scikit-Learn (SVM/Random Forest) 以及 60 維幾何歸一化。



## 安裝與運行指南 (Installation & Quick Start)：

```bash
git clone https://github.com/YourUsername/SignPlay-ASL.git
```
```bash
cd SignPlay-ASL
```
```bash
pip install -r requirements.txt
```
```bash
python game_engine.py
```
> [!NOTE]
> This SignPlayAI beta features an interactive user-hint system and runs on a custom AI model independently trained on more than 2500 gesture samples. The main Project is trained on 30,000+ datasets
## 操作說明 (Controls)：

主菜單下按 1-4 切換模式。

遊戲內擺出對應手勢進行識別。

Spacebar 跳過當前題目，ESC 返回主菜單。

youtube video demo: https://youtu.be/kQiwyxtRBqw

## 未來擴展與隱私保障 (Future Roadmap & Privacy)：
提及匿名排行榜、全句手語翻譯（SLT）流利度評估，以及純邊緣端內存即時銷毀的校園安全機制（這能極大拔高項目的學術深度）。
