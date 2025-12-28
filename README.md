# **Radarm 本地部署与使用指南 🚀**

Radarm 是一个集成机器学习、多专家协作（Multi-Agent）和数据库连接的现代数据分析平台。

本指南将帮助你在本地环境（Windows/macOS/Linux）快速部署 Radarm。

## **🛠️ 第一步：环境准备**

在开始之前，请确保你的电脑上已经安装了以下基础软件：

1. **Node.js** (用于前端): [下载地址](https://nodejs.org/) (建议版本 v16 或更高)  
2. **Python** (用于后端): [下载地址](https://www.python.org/) (建议版本 3.8 或更高)

## **🖥️ 第二步：前端部署 (React)**

前端负责用户界面和交互。

1. 进入前端目录  
   打开终端（Terminal 或 CMD），进入项目文件夹：  
   cd radarm

2. 安装依赖  
   一次性安装所有前端库（包括 React, Tailwind CSS, 图标库等）：  
   npm install  
   \# 如果缺少图标库，单独运行:  
   npm install lucide-react  
   \# 安装 Tailwind CSS (确保是 v3 版本以兼容当前配置)  
   npm install \-D tailwindcss@3.4.17 postcss autoprefixer

3. 初始化样式 (仅首次运行需要)  
   如果你还没有配置 Tailwind，运行：  
   npx tailwindcss init \-p

   *确保 tailwind.config.js 和 src/index.css 配置正确（参考之前的教程）。*  
4. **启动前端**  
   npm start

   浏览器会自动打开 http://localhost:3000。此时你会看到 Radarm 的界面，但会提示连接失败，因为后端还没启动。

## **🧠 第三步：后端部署 (Python FastAPI)**

后端负责数据处理、AI 交互和机器学习运算。

1. 准备后端文件  
   确保你的根目录下有以下核心文件：  
   * backend.py (主程序 / API 路由)  
   * ml\_engine.py (机器学习引擎)  
   * engine\_agent\_single.py (单模型 Agent 引擎)  
   * engine\_agent\_multi.py (多专家混合 Agent 引擎)  
   * agent\_tool\_workflow.py (工具调用型 Agent 工作流，支持多步工具循环)  
   * analysis\_engine.py (统计/建模分析引擎，确定性算法)  
   * engine\_report.py / engine\_report\_v2.py (报告生成引擎：多阶段、多模型)  
   * workflow\_single\_chat.py / workflow\_multi\_chat.py / workflow\_report.py (工作流入口薄封装)  
   * session\_store.py (本地持久化：session 数据与元数据)  
   * action\_engine.py (Action Pipeline：结构化数据清洗/变换)  
   * requirements.txt (依赖表)  
2. 安装 Python 依赖  
   在终端中运行（建议在新的终端窗口中操作，不要关闭前端窗口）：  
   pip install \-r requirements.txt

   *(注：下载慢可以使用国内源，例如加上 \-i https://pypi.tuna.tsinghua.edu.cn/simple)*  
3. **启动后端服务**  
   python backend.py

   当看到 Uvicorn running on http://0.0.0.0:8000 时，说明后端启动成功。

## **🎮 第四步：使用说明**

现在前端和后端都已启动，回到浏览器 (http://localhost:3000) 开始使用。

### **1\. 配置 API Key 🔑**

* 点击左下角的 **设置图标 (Settings)**。  
* 支持配置 **1 个 DeepSeek API Key + 1 个智谱 API Key + 1 个千问 API Key（全局）**，并在所有任务窗口通用（新建任务无需重复配置）。  
  * DeepSeek Key 会自动用于 **DeepSeek-A / DeepSeek-B / DeepSeek-C**（角色槽位）

### **2\. 导入数据 📂**

* 点击顶部的 **“导入文件”**。  
* 支持格式：.csv, .xlsx (Excel), .json, .parquet。  
* 或者点击 **“连接数据库”**，输入 MySQL/PostgreSQL/SQLite 的连接信息。
* 导入成功后，系统会自动进行一次 **导入后数据预检**：生成 **数据概览表+缺失率图**，并给出 **清洗/变换建议（需确认）** 与 **分析建议（下一步）**。

### **3\. 选择运行模式 ⚡**

在**输入栏**处切换模式（设置面板只负责 API/模型偏好配置）：

* **Ask（普通问答，单API）**：  
  * 支持 **联网搜索** 与 **上传图片**  
  * **图片理解**：默认启用视觉模型（**GLM-4V** / **Qwen-VL / Qwen-Omni**，如 `qwen-omni-turbo` / `qwen3-omni-flash`）将图片转为“文字描述”再注入对话上下文；需要配置 **智谱 Key 或千问 Key**，否则会自动降级为 **OCR/仅文件名**  
  * 支持 **@列名** 引用数据字段（用于更精准提问）
* **Agent（单模型）**：  
  * 面向数据分析/代码执行闭环（含 Action Pipeline）  
  * 支持 **联网搜索** 与 **上传图片**  
  * **图片理解**：同 Ask（由视觉模型先“看图”生成文字描述，再交给当前 Agent 模型）  
  * 支持 **@列名** 引用数据字段  
* **Agent（多专家）** 🔥：  
  * Planner / Executor / Verifier 三角色闭环（可在输入栏为每个角色选择 provider+model）  
  * 适合：复杂建模、低翻车闭环推理  
  * 支持 **联网搜索** 与 **上传图片**  
  * **图片理解**：同 Ask（由视觉模型先“看图”生成文字描述，再交给多专家工作流）  
  * 支持 **@列名** 引用数据字段

### **4\. 核心功能演示 💡**

* **数据清洗（Action 模式）**："删除所有含有空值的行，并把 score 列标准化。"  
  * 现在清洗会先给出“清洗/变换建议（需确认）”，点击 **应用所选操作** 后才会真正改数据  
  * **Agent 模式智能续跑**：在 Agent 模式下，如果清洗后用户意图包含分析需求（如“清洗后做相关性分析”），系统会自动在应用清洗后继续执行分析，并返回表格+图表+总结（无需再次输入）  
  * 支持 **撤销/重做**（顶部按钮）  
* **工具面板（M2）**：顶部点击“滑杆”图标打开  
  * **列属性侧栏/元数据**：点击表格列头进入“列属性”，可编辑变量标签、度量类型、值标签、缺失码  
  * **历史栈可视化**：进入“历史”查看每一步 Action，支持点击任意步骤 **跳转回放**（相当于 time-travel）  
  * **菜单化清洗**：进入“清洗菜单”，用按钮一键执行常用清洗（同样可撤销/重做/跳转）  
  * **统计/建模分析（A1）**：进入“分析”，一键运行 **表格（可复制）+ DeepSeek 文本解释 + 图表**  
* **绘图**："画一个散点图展示 age 和 score 的关系。"  
  * *点击生成的图片可以全屏预览和下载。*  
* **机器学习 (预测)**："建立一个回归模型，根据学习时间预测成绩。"  
  * *后台会自动调用 ml\_engine 进行训练和评估。*  
* **一键生成数据分析报告（Report v2）**：  
  * 切换到左侧 **“文件”** 图标（报告页）。  
  * 左侧可填写 **分析需求（可用 @列名）**、选择 **字段子集**、设置 **抽样行数**，并为 **规划/洞察/成文/审校** 4 个阶段选择不同的 provider+model（DeepSeek / 智谱 / 千问 均可）。  
  * 点击 **“生成新报告”**：会在同一任务下生成多份报告，并在“报告列表”中可预览切换。  
  * 点击 **“覆盖当前报告”**：在当前选中的报告上重新生成（便于快速迭代）。  
  * 报告会同时生成 **图表（PNG）+ 图表数据（CSV）**，并支持导出：
    - **导出MD**：导出 Markdown 报告正文
    - **导出ZIP**：导出 report.md + 图表PNG + 图表数据CSV + manifest.json（全量产物）

## **❓ 常见问题排查**

Q: 上传文件报错 "500 Internal Server Error"?  
A: 检查后端终端的报错信息。通常是因为文件编码问题（CSV 尽量用 UTF-8）或缺少 openpyxl 等库。确保你已经运行了 pip install \-r requirements.txt。  
Q: Agent 多专家模式卡住不动？  
A: 该模式需要进行多轮 AI 交互和代码执行，速度较慢（常见 30-60 秒）。你可以在输入栏点击 **停止** 中断等待；同时可展开“思考日志”查看进度，若长期无响应请检查后端终端是否有网络超时。  
Q: 清洗后自动续跑分析是如何触发的？  
A: 在 Agent 模式下，如果用户输入包含清洗意图+分析意图（如“删除空值后做相关性分析”），系统会在应用清洗后自动检测并续跑分析。你也可以在清洗建议阶段明确表达分析需求，系统会记住并在应用后自动执行。  
Q: 图表无法显示中文？  
A: 后端已默认配置 SimHei (黑体) 和 Microsoft YaHei。如果你的服务器是 Linux (如 Ubuntu) 且未安装中文字体，图表中文可能会显示为方框。解决方法是安装中文字体包：sudo apt-get install fonts-wqy-zenhei。

## **💾 本地持久化说明**

- Radarm 会把每个任务窗口（session）的原始数据与清洗历史持久化到 `radarm_data/sessions/` 下（自动创建）
- 重启后端后，前端会在切换任务时自动从后端同步并恢复数据与历史
- 如需彻底清空本地数据，可删除 `radarm_data/` 目录（已加入 `.gitignore`）

## **🗂️ 对话产物（图表）存放**

- Ask/Agent/分析 面板产生的图表统一保存到 `out/{session_id}/`，前端通过接口读取展示
- **报告（Report v2）** 会把产物保存到 `out/{session_id}/reports/{report_id}/`（包含 report.md、图表PNG、图表数据CSV、manifest.json）
- **重置任务** 或 **删除任务窗口** 会自动清理对应的 `out/{session_id}/` 目录（含 reports 子目录）

## **📊 统计/建模分析（首批支持）**

工具面板 → **分析**，支持以下算法（结果会以：表格 + 解释 + 图表 输出到聊天流）：

- **描述性分析**：数据概览、频数分析、列联(交叉)+卡方、描述统计、分类汇总、正态性检验
- **差异性分析**：单样本/独立样本/配对样本 T 检验、单因素 ANOVA、卡方检验、非参数检验（Mann-Whitney / Kruskal-Wallis / Friedman）
- **相关/回归/建模**：Pearson/Spearman/Kendall 相关、线性回归（OLS，带近似 p 值）、逻辑回归（优先 statsmodels，fallback sklearn）、PCA、KMeans

## **🧩 DeepSeek-A/B/C 的高级配置（可选）**

默认情况下，DeepSeek-A/B/C 指向 DeepSeek 官方接口。你可以用环境变量覆盖 URL 与默认模型名：

- `DEEPSEEK_BASE_URL`：统一基础 URL（默认 `https://api.deepseek.com/chat/completions`）
- `DEEPSEEK_A_URL` / `DEEPSEEK_B_URL` / `DEEPSEEK_C_URL`：分别覆盖 A/B/C 的 URL
- `DEEPSEEK_A_MODEL` / `DEEPSEEK_B_MODEL` / `DEEPSEEK_C_MODEL`：分别覆盖 A/B/C 的模型名（默认 `deepseek-reasoner`）