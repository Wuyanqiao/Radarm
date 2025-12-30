# **Radarm 项目全方位改进与生产级优化方案**

## **0\. 项目背景与技术决策确认 (Project Context)**

基于前期的调研，本项目已明确以下关键技术路线：

1. **基础设施**：单台 Linux 服务器部署。架构需轻量化，避免过度复杂的 K8s 集群维护，利用 Docker Compose 进行编排。  
2. **计算引擎**：  
   * **小数据 (\<100MB)**：使用 Pandas 内存计算，极速响应。  
   * **大数据 (1GB+)**：引入 **Dask**。Dask 完美兼容 Pandas API，支持单机多核并行计算和核外计算（Out-of-Core），在单机处理大文件时比 Spark 更轻量且高效。  
3. **核心交互理念**：**“透明盒”模式 (Transparent Box)**。  
   * 面向“小白”：提供傻瓜式 Chat 引导和参数表单。  
   * 面向“学习者”：实时展示 AI 生成的 Python 代码（Notebook 风格），并允许用户直接修改代码并重新运行。  
4. **模型策略**：调用高性能公网 API (GPT-4 / DeepSeek-V3 / Qwen-Max) 以保证代码生成的准确率，本地专注于业务逻辑和数据安全。

## **1\. 现状诊断与核心痛点**

基于现有代码库（backend, action\_engine, workflow\_\*），项目目前的形态偏向于一个“基于 LLM 的脚本执行器”。要达到生产级别并对标 SPSSPRO，存在以下核心短板：

1. **安全性高危**：代码似乎直接在宿主机环境运行（execute\_code），一旦 Agent 生成恶意代码（如 os.system('rm \-rf /')），服务器将面临毁灭性打击。  
2. **状态管理脆弱**：session\_store.py 看起来是内存级别的存储，服务重启数据即丢失，无法支持长周期的分析任务。  
3. **计算阻塞**：复杂的 ML 训练或大数据处理若在主线程运行，会阻塞 Web 服务，导致前端超时。  
4. **交互单一**：缺乏“代码可见性”。用户无法验证 AI 的分析逻辑，也无法通过修改 K 值、删除异常点等微调分析结果。  
5. **算法深度不足**：目前的 ml\_engine 缺乏针对大数据的优化和专业的自动化特征工程。

## **2\. 架构重构（Foundation）**

### **2.1 单机容器化沙箱（Docker Sandbox）**

**优先级：最高 (P0)**

* **架构设计**：  
  * 采用 **Session-based Container** 模式。  
  * 当用户开始一个新的分析项目时，后端使用 Docker SDK (docker.from\_env()) 启动一个独立的 Python 容器。  
  * **挂载策略**：将宿主机的 ./data/{user\_id}/{project\_id} 目录挂载到容器的 /workspace。  
  * **生命周期管理**：设置 Redis 过期键监听，当用户 30 分钟无操作，自动销毁容器，释放内存。  
* **安全限制**：  
  * network\_mode="none"（或仅允许访问公网 API 白名单）。  
  * mem\_limit="4g"，cpu\_quota 限制，防止单用户耗尽服务器资源。

### **2.2 混合计算引擎 (Hybrid Engine)**

**优先级：高 (P1)**

* **智能分流**：  
  * 后端在读取文件头时判断文件大小。  
  * **If file\_size \< 100MB**: Agent 生成标准 Pandas 代码。  
  * **If file\_size \> 100MB**: Agent 自动切换 System Prompt，生成 **Dask** 代码（例如 import dask.dataframe as dd）。  
* **异步任务队列**：  
  * 使用 **Celery \+ Redis**。  
  * 长耗时任务（如 1GB 数据的 groupby 或模型训练）通过 WebSocket 实时推送进度条给前端，避免 HTTP 超时。

### **2.3 持久化层改造**

**优先级：高 (P1)**

* **SQLite / PostgreSQL**：考虑到单机部署，如果并发不高，Docker 化的 PostgreSQL 是最佳选择，存储 User, Project, ChatHistory。  
* **文件系统**：  
  * 用户上传的数据、生成的图表、训练好的模型 (.pkl) 统一存储在本地文件系统，通过 Nginx 映射访问，或者集成 MinIO（如果未来考虑迁移到 S3）。

## **3\. 核心分析能力增强（The "Pro" Capabilities）**

### **3.1 智能数据预处理管道 (Smart ETL)**

* **自动类型推断**：区分名义变量、定序变量、连续变量。  
* **Dask 集成**：对于大文件，使用 Dask 实现并行清洗。  
* **智能清洗**：  
  * 基于 LLM 的语义清洗（例如把 “男”、“Man”、“M” 统一为 “Male”）。  
  * 异常值检测（3-Sigma, IQR, Isolation Forest）。

### **3.2 引入“统计专家系统”**

* **改进方案**：  
  * 建立一个严格的 **StatLib**（统计库封装）。  
  * **假设检验自动化**：用户说“对比两组数据”，系统自动进行正态性检验 \-\> 方差齐性检验 \-\> 自动选择 T 检验或曼-惠特尼 U 检验。  
  * **因子分析/PCA**：提供完整的解释方差表和碎石图。

### **3.3 AutoML 集成**

* **集成框架**：集成 FLAML (轻量级，微软出品) 或 AutoGluon。  
* **可解释性 (XAI)**：  
  * 不仅仅给出预测结果，必须生成 **SHAP 值图**，告诉用户“为什么这个客户流失风险高？因为他的‘月消费’下降了”。

## **4\. 交互体验升级：打造“透明盒”学习型 UI**

这是解决你“用户想学代码”痛点的核心。

### **4.1 双模态界面 (Dual-Mode Interface)**

* **左侧：Chat & Guide**  
  * 自然语言对话。  
  * 参数配置表单（例如：滑块调整聚类数 K=3）。  
* **右侧：Live Code Notebook**  
  * **只读模式**：AI 生成代码后，代码块自动高亮显示，并在下方展示运行结果（表格、图表）。  
  * **编辑模式**：用户点击“编辑代码”，进入 Monaco Editor（VS Code 同款编辑器）。用户可以修改 df.dropna() 为 df.fillna(0)，点击“重新运行”，立即看到新结果。  
  * **AI 解释**：鼠标悬停在某行代码上，显示 AI 生成的中文解释（例如：StandardScaler() \-\> "这里正在对数据进行标准化处理，为了消除量纲影响..."）。

### **4.2 电子表格交互 (Spreadsheet UI)**

* 集成 **Ag-Grid** (社区版免费且强大)。  
* 支持懒加载（Infinite Scroll），即使是 1GB 数据，前端只请求当前视口的 100 行数据，流畅展示。

### **4.3 交互式可视化**

* 使用 **ECharts** 或 **Plotly.js** 替代静态图片。  
* 图表支持缩放、筛选、导出为 PNG/SVG。

## **5\. Agent 智能层进化**

### **5.1 提示词工程 (Prompt Engineering)**

* **针对大数据的 Prompt**：  
  * "You are a Dask expert. The dataset is large (2GB). Use dask.dataframe instead of pandas. Do not call .compute() until necessary."  
* **针对教学的 Prompt**：  
  * "Generate code with detailed Chinese comments explaining the statistical logic." (生成带有详细中文注释的代码，解释统计学逻辑)。

### **5.2 动态规划与自我修正 (Self-Correction)**

* 当代码执行报错（例如 Dask 内存溢出），将 Traceback 回传给 Agent，Agent 自动尝试优化代码（例如增加 chunksize）。

## **6\. 详细实施路线图 (Roadmap)**

### **第一阶段：稳固根基 (1-2 个月)**

1. **环境容器化**：编写 docker-compose.yml，包含 App, Worker (Celery), Redis, Postgres。  
2. **Docker 沙箱开发**：实现 SandboxService，负责启动容器、执行代码、获取结果。  
3. **Dask 整合**：验证 Dask 在 Docker 环境下的单机多进程运行稳定性。

### **第二阶段：交互升级 (2-3 个月)**

1. **前端重构**：引入 Monaco Editor 和 Ag-Grid。实现“左对话，右代码”的布局。  
2. **WebSocket 管道**：打通 后端执行日志 \-\> 前端控制台 的实时流。  
3. **可视化升级**：从返回 PNG 图片改为返回 ECharts JSON 配置。

### **第三阶段：AI 深度集成 (持续迭代)**

1. **AutoML 模块**：引入自动模型训练功能。  
2. **智能报告生成**：生成 PDF 研报。  
3. **统计专家库**：完善各类假设检验的自动路由逻辑。

## **7\. 技术栈推荐 (Updated)**

| 模块 | 推荐技术 | 理由 |
| :---- | :---- | :---- |
| **Backend** | Python, FastAPI | 高性能，原生支持异步，OpenAPI 文档友好 |
| **Large Data Engine** | **Dask** | 单机处理 1GB+ CSV 的最佳 Python 原生方案，无需 Spark JVM 依赖 |
| **Small Data Engine** | Pandas | 灵活、快速 |
| **Task Queue** | Celery \+ Redis | 异步任务调度 |
| **Sandbox** | **Docker SDK (Python)** | 简单可靠的容器管理 |
| **Database** | PostgreSQL | 稳定可靠的关系型数据库 |
| **Frontend** | React \+ **Monaco Editor** | 实现 IDE 级别的代码编辑体验 |
| **Data Grid** | **Ag-Grid** | 处理百万行数据展示不卡顿 |
| **Charts** | ECharts | 百度开源，图表类型丰富，交互性好 |
| **LLM API** | OpenAI / DeepSeek | 智谱/DeepSeek 性价比极高，适合高频代码生成 |

通过这套方案，Radarm 不仅是一个工具，更是一个\*\*“AI 驱动的数据分析教学平台”\*\*，这完美契合了小白用户渴望成长的心理，形成了极强的差异化竞争壁垒。