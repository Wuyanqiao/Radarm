import React, { useState, useEffect, useRef } from 'react';
import { Upload, MessageSquare, Play, BarChart2, FileText, Settings, Database, X, Loader2, Terminal, Zap, Trash2, Copy, Server, Plus, Layers, PanelLeftClose, PanelLeft, Users, ChevronDown, ChevronUp, ChevronLeft, ChevronRight, Printer, FileDown, RotateCcw, RotateCw, SlidersHorizontal, AtSign, Globe, Image as ImageIcon, Eye, Command } from 'lucide-react';

const uuid = () => Math.random().toString(36).substr(2, 9);
const API_BASE = 'http://localhost:8000';
const PROVIDER_LABELS = {
  deepseekA: 'DeepSeek-A',
  deepseekB: 'DeepSeek-B',
  deepseekC: 'DeepSeek-C',
  zhipu: 'Zhipu',
  qwen: 'Qwen'
};

const MODEL_OPTIONS = {
  deepseek: ['deepseek-chat', 'deepseek-reasoner'],
  zhipu: ['glm-4.7', 'glm-4.6', 'glm-4.5'],
  qwen: ['qwen-max', 'qwen-long-latest', 'qwen-coder-plus-latest', 'qwen-omni-turbo', 'qwen3-omni-flash']
};

// 视觉理解模型选项
const VISION_MODEL_OPTIONS = {
  zhipu: [
    { value: 'glm-4v', label: 'GLM-4V（默认）' },
    { value: 'glm-4v-0524', label: 'GLM-4V-0524' },
  ],
  qwen: [
    { value: 'qwen-vl-plus', label: 'Qwen-VL-Plus（默认）' },
    { value: 'qwen-vl-max', label: 'Qwen-VL-Max' },
    { value: 'qwen-omni-turbo', label: 'Qwen-Omni-Turbo' },
    { value: 'qwen3-omni-flash', label: 'Qwen3-Omni-Flash' },
  ],
  auto: [
    { value: '', label: '自动选择（默认）' },
    { value: 'glm-4v', label: 'GLM-4V' },
    { value: 'qwen-vl-plus', label: 'Qwen-VL-Plus' },
    { value: 'qwen-omni-turbo', label: 'Qwen-Omni-Turbo' },
    { value: 'qwen3-omni-flash', label: 'Qwen3-Omni-Flash' },
  ]
};

// Provider 列表（全局配置 1 个 DeepSeek Key + 1 个智谱 Key + 1 个千问 Key；DeepSeek 仍保留 A/B/C 作为角色槽位）
const allProviderIds = ['deepseekA','deepseekB','deepseekC','zhipu','qwen'];

const createNewSession = (index) => ({
  id: uuid(),
  name: `任务 ${index + 1}`,
  data: [],
  dataMeta: { rows: 0, cols: 0, filename: "未导入数据" },
  history: { cursor: 0, total: 0, can_undo: false, can_redo: false },
  historyStack: [],
  meta: null, // 后端返回：{ columns: { [colName]: {label, measure, value_labels, missing_codes, _orig} }, maps: {orig_to_cur, cur_to_orig} }
  dataProfile: null,
  messages: [{ role: 'ai', content: 'Radarm 准备就绪。' }],
  // API Keys（历史字段：不再按 session 配置；现在使用全局 DeepSeek Key 自动映射到 deepseekA/B/C）
  apiKeys: { deepseekA: '', deepseekB: '', deepseekC: '', zhipu: '', qwen: '' },
  // 输入栏切换：ask / agent_single / agent_multi
  runMode: 'ask',
  // 各模式模型选择（输入栏可切换）
  modelPrefs: {
    ask: { provider: 'deepseekA', model: 'deepseek-chat' },
    agent_single: { provider: 'deepseekA', model: 'deepseek-reasoner' },
    agent_multi: {
      planner: { provider: 'deepseekA', model: 'deepseek-reasoner' },
      executor: { provider: 'deepseekB', model: 'deepseek-reasoner' },
      verifier: { provider: 'deepseekC', model: 'deepseek-reasoner' },
    }
  },
  askWebSearch: false,
  askImages: [],
  // 视觉理解（GLM-4V / Qwen-VL）：默认启用；auto=优先智谱，无则千问
  visionEnabled: true,
  visionProvider: 'qwen', // 固定使用 qwen
  visionModel: 'qwen-omni-turbo', // 固定使用 qwen-omni-turbo
  inputValue: '',
  isAnalyzing: false,
  thinkingStartedAt: null,
  lastThinkMs: null,
  isUploading: false,
  isGeneratingReport: false,
  generatingReportId: null, // 正在生成的报告 ID（用于取消）
  // 报告（v2）：多份报告列表 + 当前预览
  reports: [], // [{report_id,title,created_at,...}]
  activeReportId: null,
  reportContent: '',
  reportLog: '', 
  reportCharts: [],
  reportTables: [],
  reportDraft: {
    userRequest: '',
    selectedColumns: [],
    sampleRows: '',
    stages: {
      planner: { provider: 'deepseekA', model: 'deepseek-reasoner' },
      analyst: { provider: 'deepseekB', model: 'deepseek-reasoner' },
      writer: { provider: 'deepseekA', model: 'deepseek-reasoner' },
      reviewer: { provider: 'deepseekC', model: 'deepseek-reasoner' },
    }
  },
  activeTab: 'data'
});

// 统计/建模：算法目录（可搜索/树形菜单） —— 仅首批已落地算法标记为 ready
const ANALYSIS_ALGOS = [
  // 描述性分析
  { id: 'overview', name: '数据概览', categoryPath: ['描述性分析'], status: 'ready', analysis: 'overview', fields: [] },
  { id: 'frequency', name: '频数分析', categoryPath: ['描述性分析'], status: 'ready', analysis: 'frequency', fields: [
    { key: 'column', label: '列', type: 'column', required: true, fallbackSelected: true },
    { key: 'top_n', label: 'Top N', type: 'number', required: false, default: 30 },
  ]},
  { id: 'crosstab', name: '列联(交叉)分析', categoryPath: ['描述性分析'], status: 'ready', analysis: 'crosstab', fields: [
    { key: 'row', label: '行变量', type: 'column', required: true },
    { key: 'col', label: '列变量', type: 'column', required: true },
  ]},
  { id: 'descriptive', name: '描述性统计', categoryPath: ['描述性分析'], status: 'ready', analysis: 'descriptive', fields: [
    { key: 'columns', label: '列（可多选，留空=全数值列）', type: 'columns', required: false },
  ]},
  { id: 'group_summary', name: '分类汇总', categoryPath: ['描述性分析'], status: 'ready', analysis: 'group_summary', fields: [
    { key: 'group_by', label: '分组列', type: 'column', required: true },
    { key: 'metric', label: '指标列', type: 'column', required: true },
    { key: 'agg', label: '聚合', type: 'select', required: true, default: 'mean', options: ['mean', 'median', 'sum', 'count'] },
  ]},
  { id: 'normality', name: '正态性检验', categoryPath: ['描述性分析'], status: 'ready', analysis: 'normality', fields: [
    { key: 'column', label: '列', type: 'column', required: true, fallbackSelected: true },
    { key: 'method', label: '方法', type: 'select', required: true, default: 'auto', options: ['auto', 'shapiro', 'normaltest', 'jarque_bera'] },
  ]},

  // 差异性分析（参数检验）
  { id: 'ttest_one', name: '单样本T检验', categoryPath: ['差异性分析', '参数检验'], status: 'ready', analysis: 'ttest', fixedParams: { ttype: 'one_sample' }, fields: [
    { key: 'y', label: 'y', type: 'column', required: true, fallbackSelected: true },
    { key: 'mu', label: 'mu（检验均值）', type: 'number', required: true, default: 0 },
  ]},
  { id: 'ttest_ind', name: '独立样本T检验', categoryPath: ['差异性分析', '参数检验'], status: 'ready', analysis: 'ttest', fixedParams: { ttype: 'independent' }, fields: [
    { key: 'y', label: 'y', type: 'column', required: true },
    { key: 'group_col', label: '分组列', type: 'column', required: true },
    { key: 'group_a', label: '组A（可选）', type: 'text', required: false, placeholder: '留空=自动选频数最高的两组' },
    { key: 'group_b', label: '组B（可选）', type: 'text', required: false },
  ]},
  { id: 'ttest_paired', name: '配对样本T检验', categoryPath: ['差异性分析', '参数检验'], status: 'ready', analysis: 'ttest', fixedParams: { ttype: 'paired' }, fields: [
    { key: 'y', label: 'y', type: 'column', required: true },
    { key: 'y2', label: 'y2（配对列）', type: 'column', required: true },
  ]},
  { id: 'anova_oneway', name: '单因素方差分析', categoryPath: ['差异性分析', '参数检验'], status: 'ready', analysis: 'anova', fields: [
    { key: 'y', label: 'y', type: 'column', required: true },
    { key: 'group_col', label: '分组列', type: 'column', required: true },
  ]},

  // 差异性分析（非参数检验）
  { id: 'chi_square', name: 'Pearson卡方检验', categoryPath: ['差异性分析', '非参数检验'], status: 'ready', analysis: 'chi_square', fields: [
    { key: 'row', label: 'row', type: 'column', required: true },
    { key: 'col', label: 'col', type: 'column', required: true },
  ]},
  { id: 'mann_whitney', name: '独立样本Mann-Whitney检验', categoryPath: ['差异性分析', '非参数检验'], status: 'ready', analysis: 'nonparam', fixedParams: { test: 'mann_whitney' }, fields: [
    { key: 'y', label: 'y', type: 'column', required: true },
    { key: 'group_col', label: '分组列', type: 'column', required: true },
    { key: 'group_a', label: '组A（可选）', type: 'text', required: false },
    { key: 'group_b', label: '组B（可选）', type: 'text', required: false },
  ]},
  { id: 'kruskal', name: '多独立样本Kruskal-Wallis检验', categoryPath: ['差异性分析', '非参数检验'], status: 'ready', analysis: 'nonparam', fixedParams: { test: 'kruskal' }, fields: [
    { key: 'y', label: 'y', type: 'column', required: true },
    { key: 'group_col', label: '分组列', type: 'column', required: true },
  ]},
  { id: 'friedman', name: '多配对样本Friedman检验', categoryPath: ['差异性分析', '非参数检验'], status: 'ready', analysis: 'nonparam', fixedParams: { test: 'friedman' }, fields: [
    { key: 'columns', label: '列（可多选，≥3）', type: 'columns', required: true, min: 3 },
  ]},

  // 相关性分析（拆分为三种，便于目录匹配）
  { id: 'corr_pearson', name: 'Pearson相关性分析', categoryPath: ['相关性分析'], status: 'ready', analysis: 'correlation', fixedParams: { method: 'pearson' }, fields: [
    { key: 'columns', label: '列（可多选，留空=全数值列）', type: 'columns', required: false },
  ]},
  { id: 'corr_spearman', name: 'Spearman相关性分析', categoryPath: ['相关性分析'], status: 'ready', analysis: 'correlation', fixedParams: { method: 'spearman' }, fields: [
    { key: 'columns', label: '列（可多选，留空=全数值列）', type: 'columns', required: false },
  ]},
  { id: 'corr_kendall', name: "Kendall's tau-b相关性分析", categoryPath: ['相关性分析'], status: 'ready', analysis: 'correlation', fixedParams: { method: 'kendall' }, fields: [
    { key: 'columns', label: '列（可多选，留空=全数值列）', type: 'columns', required: false },
  ]},

  // 预测模型/统计建模
  { id: 'linreg', name: '线性回归（最小二乘法）', categoryPath: ['预测模型/统计建模'], status: 'ready', analysis: 'linear_regression', fields: [
    { key: 'y', label: 'y', type: 'column', required: true },
    { key: 'x', label: 'X（可多选）', type: 'columns', required: true, min: 1 },
  ]},
  { id: 'logit', name: '逻辑回归', categoryPath: ['预测模型/统计建模'], status: 'ready', analysis: 'logistic_regression', fields: [
    { key: 'y', label: 'y（二元）', type: 'column', required: true },
    { key: 'x', label: 'X（可多选）', type: 'columns', required: true, min: 1 },
  ]},
  { id: 'pca', name: '主成分分析(PCA)', categoryPath: ['预测模型/统计建模'], status: 'ready', analysis: 'pca', fields: [
    { key: 'columns', label: '列（可多选，≥2）', type: 'columns', required: true, min: 2 },
    { key: 'n_components', label: '组件数', type: 'number', required: true, default: 2 },
  ]},
  { id: 'kmeans', name: '聚类分析(K-Means)', categoryPath: ['预测模型/统计建模'], status: 'ready', analysis: 'kmeans', fields: [
    { key: 'columns', label: '列（可多选，≥2）', type: 'columns', required: true, min: 2 },
    { key: 'k', label: 'k', type: 'number', required: true, default: 3 },
  ]},
];

export default function RadarmApp() {
  // --- 全局设置：DeepSeek API Key（跨任务窗口通用） ---
  const [globalDeepSeekKey, setGlobalDeepSeekKey] = useState(() => {
    const saved = localStorage.getItem('radarm_deepseek_key');
    return saved || '';
  });
  const [globalZhipuKey, setGlobalZhipuKey] = useState(() => {
    const saved = localStorage.getItem('radarm_zhipu_key');
    return saved || '';
  });
  const [globalQwenKey, setGlobalQwenKey] = useState(() => {
    const saved = localStorage.getItem('radarm_qwen_key');
    return saved || '';
  });

  const [sessions, setSessions] = useState(() => {
    const saved = localStorage.getItem('radarm_sessions');
    const raw = saved ? (() => { try { return JSON.parse(saved); } catch { return null; } })() : null;

    const migrateOne = (s, idx) => {
      const base = s && typeof s === 'object' ? { ...s } : createNewSession(idx);

      // mode 迁移：旧 single/expert_mixed -> 新 runMode（agent_single/agent_multi）；默认 ask
      if (!base.runMode) {
        if (base.mode === 'expert_mixed') base.runMode = 'agent_multi';
        else if (base.mode === 'single') base.runMode = 'agent_single';
        else base.runMode = 'ask';
      }

      // apiKeys 迁移：旧 deepseek/qwen/zhipu -> deepseekA/B/C；并补齐 zhipu/qwen（兼容历史）
      const oldKeys = base.apiKeys || {};
      if (!oldKeys.deepseekA && (oldKeys.deepseek || oldKeys.qwen || oldKeys.zhipu)) {
        base.apiKeys = {
          deepseekA: oldKeys.deepseek || '',
          deepseekB: oldKeys.qwen || '',
          deepseekC: oldKeys.zhipu || '',
          zhipu: oldKeys.zhipu_api || oldKeys.zhipuKey || '', // 若历史存在其他字段也尽量迁移
          qwen: oldKeys.qwen_api || oldKeys.qwenKey || '',
        };
      } else {
        base.apiKeys = {
          deepseekA: oldKeys.deepseekA || '',
          deepseekB: oldKeys.deepseekB || '',
          deepseekC: oldKeys.deepseekC || '',
          zhipu: oldKeys.zhipu || '',
          qwen: oldKeys.qwen || '',
        };
      }

      // modelPrefs 迁移：旧 primaryModel -> agent_single 的 provider
      if (!base.modelPrefs) {
        const mapOldProvider = (p) => {
          if (p === 'deepseek') return 'deepseekA';
          if (p === 'qwen') return 'deepseekB';
          if (p === 'zhipu') return 'deepseekC';
          return 'deepseekA';
        };
        const agentSingleProvider = mapOldProvider(base.primaryModel || 'deepseek');
        base.modelPrefs = {
          ask: { provider: 'deepseekA', model: 'deepseek-chat' },
          agent_single: { provider: agentSingleProvider, model: 'deepseek-reasoner' },
          agent_multi: {
            planner: { provider: 'deepseekA', model: 'deepseek-reasoner' },
            executor: { provider: 'deepseekB', model: 'deepseek-reasoner' },
            verifier: { provider: 'deepseekC', model: 'deepseek-reasoner' },
          }
        };
      } else {
        // 补齐字段
        base.modelPrefs.ask = base.modelPrefs.ask || { provider: 'deepseekA', model: 'deepseek-chat' };
        base.modelPrefs.agent_single = base.modelPrefs.agent_single || { provider: 'deepseekA', model: 'deepseek-reasoner' };
        base.modelPrefs.agent_multi = base.modelPrefs.agent_multi || {
          planner: { provider: 'deepseekA', model: 'deepseek-reasoner' },
          executor: { provider: 'deepseekB', model: 'deepseek-reasoner' },
          verifier: { provider: 'deepseekC', model: 'deepseek-reasoner' },
        };
      }

      // provider 兜底：如果历史选择了 zhipu/qwen 等，自动回落到 DeepSeek
      const _normProvider = (p, fallback) => (allProviderIds.includes(String(p || '')) ? String(p) : fallback);
      base.modelPrefs.ask.provider = _normProvider(base.modelPrefs.ask.provider, 'deepseekA');
      base.modelPrefs.agent_single.provider = _normProvider(base.modelPrefs.agent_single.provider, 'deepseekA');
      if (base.modelPrefs.agent_multi) {
        base.modelPrefs.agent_multi.planner = base.modelPrefs.agent_multi.planner || { provider: 'deepseekA', model: 'deepseek-reasoner' };
        base.modelPrefs.agent_multi.executor = base.modelPrefs.agent_multi.executor || { provider: 'deepseekB', model: 'deepseek-reasoner' };
        base.modelPrefs.agent_multi.verifier = base.modelPrefs.agent_multi.verifier || { provider: 'deepseekC', model: 'deepseek-reasoner' };
        base.modelPrefs.agent_multi.planner.provider = _normProvider(base.modelPrefs.agent_multi.planner.provider, 'deepseekA');
        base.modelPrefs.agent_multi.executor.provider = _normProvider(base.modelPrefs.agent_multi.executor.provider, 'deepseekB');
        base.modelPrefs.agent_multi.verifier.provider = _normProvider(base.modelPrefs.agent_multi.verifier.provider, 'deepseekC');
      }

      if (typeof base.askWebSearch !== 'boolean') base.askWebSearch = false;
      if (!Array.isArray(base.askImages)) base.askImages = [];
      if (typeof base.visionEnabled !== 'boolean') base.visionEnabled = true;
      // 固定使用 qwen-omni-turbo
      base.visionProvider = 'qwen';
      base.visionModel = 'qwen-omni-turbo';

      // 报告 v2 字段补齐（多份报告 + 产物）
      if (!Array.isArray(base.reports)) base.reports = [];
      if (typeof base.activeReportId === 'undefined') base.activeReportId = null;
      if (!Array.isArray(base.reportCharts)) base.reportCharts = [];
      if (!Array.isArray(base.reportTables)) base.reportTables = [];
      if (!base.reportDraft || typeof base.reportDraft !== 'object') {
        base.reportDraft = createNewSession(idx).reportDraft;
      } else {
        base.reportDraft.userRequest = String(base.reportDraft.userRequest || '');
        if (!Array.isArray(base.reportDraft.selectedColumns)) base.reportDraft.selectedColumns = [];
        if (typeof base.reportDraft.sampleRows !== 'string') base.reportDraft.sampleRows = String(base.reportDraft.sampleRows || '');
        if (!base.reportDraft.stages || typeof base.reportDraft.stages !== 'object') base.reportDraft.stages = createNewSession(idx).reportDraft.stages;
        const defStages = createNewSession(idx).reportDraft.stages;
        ['planner','analyst','writer','reviewer'].forEach(k => {
          if (!base.reportDraft.stages[k] || typeof base.reportDraft.stages[k] !== 'object') base.reportDraft.stages[k] = defStages[k];
          base.reportDraft.stages[k].provider = _normProvider(base.reportDraft.stages[k].provider || defStages[k].provider, defStages[k].provider);
          base.reportDraft.stages[k].model = base.reportDraft.stages[k].model || defStages[k].model;
        });
      }

      return base;
    };

    const arr = Array.isArray(raw) && raw.length ? raw : [createNewSession(0)];
    return arr.map((s, idx) => migrateOne(s, idx));
  });
  const [activeSessionId, setActiveSessionId] = useState(() => {
     const saved = localStorage.getItem('radarm_active_id');
     return saved || (sessions && sessions[0] ? sessions[0].id : null);
  });

  const activeSession = sessions.find(s => s.id === activeSessionId) || sessions[0];

  const [showSettings, setShowSettings] = useState(false);
  const [showDbModal, setShowDbModal] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [dbConfig, setDbConfig] = useState({ type: 'mysql', host: 'localhost', port: '3306', user: 'root', password: '', database: '', sql: 'SELECT * FROM users LIMIT 100' });
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isChatOpen, setIsChatOpen] = useState(true);
  const [logExpanded, setLogExpanded] = useState({});
  const [reportLogExpanded, setReportLogExpanded] = useState(false); 
  const [thinkingTick, setThinkingTick] = useState(Date.now());

  // M2+：工具抽屉（列属性/历史/清洗菜单/分析）
  const [toolDrawerOpen, setToolDrawerOpen] = useState(false);
  const [toolTab, setToolTab] = useState('column'); // column | history | clean | analysis
  const [selectedColumn, setSelectedColumn] = useState(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [colMetaDraft, setColMetaDraft] = useState({ label: '', measure: 'scale', valueLabelsText: '', missingCodesText: '' });
  const [cleanDraft, setCleanDraft] = useState({ missingCodes: '999', dropnaHow: 'any', castType: 'float', dedupKeep: 'first', winsorLower: '0.01', winsorUpper: '0.99', oneHotDropFirst: false, oneHotPrefixSep: '=' });
  const [reportColSearch, setReportColSearch] = useState('');
  const [analysisSearch, setAnalysisSearch] = useState('');
  const [analysisSelectedId, setAnalysisSelectedId] = useState('overview');
  const [analysisForms, setAnalysisForms] = useState({}); // { [algoId]: {field:value} }

  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);
  const chatInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const abortRef = useRef({}); // { [sessionId]: AbortController }

  // @ 引用（列名补全）
  const [mentionOpen, setMentionOpen] = useState(false);
  const [mentionQuery, setMentionQuery] = useState('');
  const [mentionStart, setMentionStart] = useState(-1);
  const [showModeModal, setShowModeModal] = useState(false);

  useEffect(() => {
    localStorage.setItem('radarm_sessions', JSON.stringify(sessions));
    localStorage.setItem('radarm_active_id', activeSessionId);
  }, [sessions, activeSessionId]);

  // 持久化全局 DeepSeek Key
  useEffect(() => {
    localStorage.setItem('radarm_deepseek_key', globalDeepSeekKey || '');
  }, [globalDeepSeekKey]);
  useEffect(() => {
    localStorage.setItem('radarm_zhipu_key', globalZhipuKey || '');
  }, [globalZhipuKey]);
  useEffect(() => {
    localStorage.setItem('radarm_qwen_key', globalQwenKey || '');
  }, [globalQwenKey]);

  // 从历史 session 的 apiKeys 中兜底迁移（只做一次：若全局 key 为空）
  useEffect(() => {
    if (globalDeepSeekKey) return;
    const found = (sessions || []).map(s => s?.apiKeys).find(k => k && (k.deepseekA || k.deepseekB || k.deepseekC));
    const v = found?.deepseekA || found?.deepseekB || found?.deepseekC || '';
    if (v) setGlobalDeepSeekKey(v);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions]);
  useEffect(() => {
    if (globalZhipuKey) return;
    const found = (sessions || []).map(s => s?.apiKeys).find(k => k && k.zhipu);
    const v = found?.zhipu || '';
    if (v) setGlobalZhipuKey(v);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions]);
  useEffect(() => {
    if (globalQwenKey) return;
    const found = (sessions || []).map(s => s?.apiKeys).find(k => k && k.qwen);
    const v = found?.qwen || '';
    if (v) setGlobalQwenKey(v);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions]);

  const getEffectiveApiKeys = () => ({
    deepseekA: globalDeepSeekKey || '',
    deepseekB: globalDeepSeekKey || '',
    deepseekC: globalDeepSeekKey || '',
    zhipu: globalZhipuKey || '',
    qwen: globalQwenKey || '',
  });

  // 思考计时：仅在当前任务 isAnalyzing=true 时刷新 UI
  useEffect(() => {
    if (!activeSession?.isAnalyzing) return;
    const timer = setInterval(() => setThinkingTick(Date.now()), 200);
    return () => clearInterval(timer);
  }, [activeSessionId, activeSession?.isAnalyzing]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeSession.messages]);

  const updateSession = (id, updates) => {
    setSessions(prev => prev.map(s => s.id === id ? { ...s, ...updates } : s));
  };

  const formatSuggestedAction = (a) => {
    const t = a?.type;
    const p = a?.params || {};
    if (t === 'replace_missing') return `将 ${p.columns?.length ? p.columns.join(', ') : '所有列'} 的 ${JSON.stringify(p.values || [])} 设为缺失`;
    if (t === 'dropna_rows') return `删除缺失行（how=${p.how || 'any'}，columns=${p.columns?.length ? p.columns.join(', ') : '全部'}）`;
    if (t === 'fillna') return `填充缺失（strategy=${p.strategy || 'value'}，columns=${p.columns?.length ? p.columns.join(', ') : '全部'}）`;
    if (t === 'cast_type') return `类型转换：${p.column} -> ${p.to}`;
    if (t === 'standardize') return `标准化：${(p.columns || []).join(', ')}`;
    if (t === 'rename_columns') return `重命名列：${JSON.stringify(p.mapping || {})}`;
    if (t === 'drop_columns') return `删除列：${(p.columns || []).join(', ')}`;
    if (t === 'deduplicate') return `去重（subset=${(p.subset || []).join(', ') || '全部'}，keep=${p.keep || 'first'}）`;
    if (t === 'trim_whitespace') return `去除空格：${p.columns?.length ? p.columns.join(', ') : '字符串列'}`;
    return JSON.stringify(a);
  };

  const providerType = (pid) => (pid && String(pid).startsWith('deepseek')) ? 'deepseek' : String(pid || '');
  const getProviderModelOptions = (pid) => MODEL_OPTIONS[providerType(pid)] || [];
  const getDefaultModelFor = (pid, runMode) => {
    const t = providerType(pid);
    if (t === 'deepseek') return runMode === 'ask' ? 'deepseek-chat' : 'deepseek-reasoner';
    if (t === 'zhipu') return 'glm-4.7';
    if (t === 'qwen') return 'qwen-max';
    return '';
  };

  const getColumnsList = (session) => {
    const s = session || activeSession;
    if (!s?.data || s.data.length === 0) return [];
    try { return Object.keys(s.data[0] || {}); } catch { return []; }
  };

  const valueLabelsToText = (obj) => {
    if (!obj || typeof obj !== 'object') return '';
    return Object.entries(obj).map(([k, v]) => `${k}=${v}`).join('\n');
  };

  const parseValueLabelsText = (text) => {
    const t = (text || '').trim();
    if (!t) return {};
    // JSON 优先
    if (t.startsWith('{')) {
      try {
        const obj = JSON.parse(t);
        if (obj && typeof obj === 'object') return obj;
      } catch {}
    }
    const out = {};
    t.split('\n').map(l => l.trim()).filter(Boolean).forEach(line => {
      const m = line.split(/=|:/);
      if (m.length >= 2) {
        const k = (m[0] || '').trim();
        const v = m.slice(1).join('=').trim();
        if (k) out[k] = v;
      }
    });
    return out;
  };

  const parseMissingCodesText = (text) => {
    const t = (text || '').trim();
    if (!t) return [];
    const tokens = t.split(/[\n,，\s]+/).map(x => x.trim()).filter(Boolean);
    return tokens.map(tok => {
      if (/^-?\d+(\.\d+)?$/.test(tok)) {
        const n = Number(tok);
        return Number.isFinite(n) ? n : tok;
      }
      return tok;
    });
  };

  // -------- 统计/建模：算法菜单树 + 参数面板（前端） --------
  const getAnalysisAlgo = (algoId) => {
    const found = (ANALYSIS_ALGOS || []).find(a => a.id === algoId);
    return found || (ANALYSIS_ALGOS && ANALYSIS_ALGOS.length ? ANALYSIS_ALGOS[0] : null);
  };

  const makeDefaultAnalysisForm = (algo) => {
    const form = {};
    (algo?.fields || []).forEach(f => {
      if (f.type === 'columns') form[f.key] = [];
      else if (f.type === 'number') form[f.key] = (f.default ?? '');
      else if (f.type === 'select') form[f.key] = (f.default ?? (f.options?.[0] ?? ''));
      else form[f.key] = (f.default ?? '');
    });
    return form;
  };

  const selectAnalysisAlgo = (algoId) => {
    setAnalysisSelectedId(algoId);
    const algo = getAnalysisAlgo(algoId);
    if (!algo) return;
    setAnalysisForms(prev => (prev?.[algoId] ? prev : { ...(prev || {}), [algoId]: makeDefaultAnalysisForm(algo) }));
  };

  const updateAnalysisFormField = (algoId, key, value) => {
    const algo = getAnalysisAlgo(algoId);
    setAnalysisForms(prev => {
      const base = (prev?.[algoId]) || (algo ? makeDefaultAnalysisForm(algo) : {});
      return { ...(prev || {}), [algoId]: { ...base, [key]: value } };
    });
  };

  const buildAnalysisTree = (items) => {
    const root = { children: {}, items: [] };
    const add = (node, path, item) => {
      // 注意：path 可能是 [] / undefined。若是空数组，视为叶子节点，避免无限递归。
      if (Array.isArray(path) && path.length === 0) { node.items.push(item); return; }
      const p = Array.isArray(path) && path.length ? path : ['其他'];
      if (p.length === 0) { node.items.push(item); return; }
      const head = p[0];
      node.children[head] = node.children[head] || { children: {}, items: [] };
      add(node.children[head], p.slice(1), item);
    };
    (items || []).forEach(it => add(root, it.categoryPath, it));
    return root;
  };

  const countAnalysisTreeItems = (node) => {
    if (!node) return 0;
    const childCount = Object.values(node.children || {}).reduce((sum, n) => sum + countAnalysisTreeItems(n), 0);
    return (node.items?.length || 0) + childCount;
  };

  const runSelectedAnalysisAlgo = () => {
    const algo = getAnalysisAlgo(analysisSelectedId);
    if (!algo) return;
    if (algo.status !== 'ready') return alert("该算法暂未开放。");

    const cols = getColumnsList(activeSession);
    const rawForm = analysisForms?.[algo.id] || makeDefaultAnalysisForm(algo);
    const params = {};
    const errors = [];

    (algo.fields || []).forEach(f => {
      let v = rawForm[f.key];

      // fallback: 使用已选列
      if (f.type === 'column') {
        if ((!v || v === '') && f.fallbackSelected && selectedColumn && cols.includes(selectedColumn)) v = selectedColumn;
        if (f.required && (!v || v === '')) errors.push(`请先选择：${f.label}`);
        if (v) params[f.key] = v;
      } else if (f.type === 'columns') {
        const arr = Array.isArray(v) ? v : [];
        if (f.required && arr.length === 0) errors.push(`请先选择：${f.label}`);
        if (f.min && arr.length < f.min) errors.push(`${f.label} 至少需要选择 ${f.min} 个`);
        if (arr.length) params[f.key] = arr;
      } else if (f.type === 'number') {
        const num = Number(v);
        if (!Number.isFinite(num)) {
          if (f.required) errors.push(`请输入数字：${f.label}`);
        } else {
          params[f.key] = num;
        }
      } else if (f.type === 'select') {
        if (f.required && (!v || v === '')) errors.push(`请先选择：${f.label}`);
        if (v !== undefined && v !== null && v !== '') params[f.key] = v;
      } else if (f.type === 'text') {
        if (v !== undefined && v !== null && String(v).trim() !== '') params[f.key] = String(v).trim();
      }
    });

    if (errors.length) return alert(errors[0]);
    const finalParams = { ...(algo.fixedParams || {}), ...params };
    runAnalysis(activeSession.id, algo.analysis, finalParams, `【分析】${algo.name}`);
  };

  const openColumnTools = (col) => {
    const m = activeSession?.meta?.columns?.[col] || {};
    setSelectedColumn(col);
    setRenameDraft(col);
    setColMetaDraft({
      label: m.label || '',
      measure: m.measure || 'scale',
      valueLabelsText: valueLabelsToText(m.value_labels || {}),
      missingCodesText: Array.isArray(m.missing_codes) ? m.missing_codes.join(', ') : '',
    });
    setToolTab('column');
    setToolDrawerOpen(true);
  };

  const applyActionsDirect = async (sessionId, actions, note) => {
    if (!actions || actions.length === 0) return;
    try {
      const response = await fetch(`${API_BASE}/apply_actions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, actions })
      });
      const res = await response.json();
      if (!response.ok) throw new Error(res.detail || '应用失败');

      setSessions(prev => prev.map(item => {
        if (item.id !== sessionId) return item;
        const msg = { role: 'ai', content: note || res.message || '已应用操作。', code: res.generated_code, result: null, image: null, log: '' };
        return {
          ...item,
          messages: [...item.messages, msg],
          data: res.new_data_preview || item.data,
          dataMeta: { ...item.dataMeta, rows: res.rows ?? item.dataMeta.rows, cols: res.cols ?? item.dataMeta.cols },
          history: res.history || item.history,
          historyStack: res.history_stack || item.historyStack,
          meta: res.meta || item.meta,
          dataProfile: res.profile || item.dataProfile,
        };
      }));
    } catch (e) {
      alert("应用失败：" + e.message);
    }
  };

  const handleSaveColumnMeta = async (sessionId) => {
    if (!selectedColumn) return;
    try {
      const payload = {
        session_id: sessionId,
        column: selectedColumn,
        label: colMetaDraft.label || '',
        measure: colMetaDraft.measure || 'scale',
        value_labels: parseValueLabelsText(colMetaDraft.valueLabelsText),
        missing_codes: parseMissingCodesText(colMetaDraft.missingCodesText),
      };
      const response = await fetch(`${API_BASE}/update_column_meta`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const res = await response.json();
      if (!response.ok) throw new Error(res.detail || '保存失败');
      updateSession(sessionId, { meta: res.meta || activeSession.meta, history: res.history || activeSession.history, historyStack: res.history_stack || activeSession.historyStack });
      // 轻量提示
      setSessions(prev => prev.map(item => item.id === sessionId ? { ...item, messages: [...item.messages, { role: 'ai', content: res.message || '已保存列元数据。' }] } : item));
    } catch (e) {
      alert("保存失败：" + e.message);
    }
  };

  const handleJumpToCursor = async (sessionId, cursor) => {
    try {
      const response = await fetch(`${API_BASE}/set_cursor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, cursor })
      });
      const res = await response.json();
      if (!response.ok) throw new Error(res.detail || '跳转失败');
      updateSession(sessionId, {
        data: res.new_data_preview || activeSession.data,
        dataMeta: { ...activeSession.dataMeta, rows: res.rows ?? activeSession.dataMeta.rows, cols: res.cols ?? activeSession.dataMeta.cols },
        history: res.history || activeSession.history,
        historyStack: res.history_stack || activeSession.historyStack,
        meta: res.meta || activeSession.meta,
        dataProfile: res.profile || activeSession.dataProfile,
      });
      setSessions(prev => prev.map(item => item.id === sessionId ? { ...item, messages: [...item.messages, { role: 'ai', content: res.message || '已跳转步骤。' }] } : item));
    } catch (e) {
      alert("跳转失败：" + e.message);
    }
  };

  const runAnalysis = async (sessionId, analysis, params, userLabel) => {
    const s = sessions.find(x => x.id === sessionId);
    if (!s) return;
    if (!s.data || s.data.length === 0) { alert("请先导入数据"); return; }

    const ms = s.modelPrefs?.agent_single || { provider: 'deepseekA', model: 'deepseek-reasoner' };
    const provider = ms.provider || 'deepseekA';
    const model = ms.model || getDefaultModelFor(provider, 'agent_single');
    const effKeys = getEffectiveApiKeys();
    if (!effKeys?.[provider]) { alert(`请先配置 ${PROVIDER_LABELS[provider] || provider} 的 API Key`); setShowSettings(true); return; }

    const startedAt = Date.now();
    const label = userLabel || `【分析】${analysis}`;
    updateSession(sessionId, { messages: [...s.messages, { role: 'user', content: label }], isAnalyzing: true, thinkingStartedAt: startedAt });

    try {
      const controller = new AbortController();
      abortRef.current[sessionId] = controller;
      const response = await fetch(`${API_BASE}/analysis_run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          analysis,
          params: params || {},
          apiKeys: getEffectiveApiKeys(),
          provider,
          model,
          explain: true,
        }),
        signal: controller.signal
      });
      const res = await response.json();
      const durationMs = Date.now() - startedAt;
      if (!response.ok) throw new Error(res.detail || '分析失败');

      setSessions(prev => prev.map(item => {
        if (item.id !== sessionId) return item;
        const images = (res.charts || []).map(c => c.path).filter(Boolean);
        const newMsg = {
          role: 'ai',
          content: res.reply || `✅ 已完成：${res.title || analysis}`,
          tables: res.tables || [],
          images,
          image: res.image || (images.length ? images[0] : null),
          code: null,
          result: null,
          log: res.process_log || '',
          needsConfirmation: false,
          actions: null,
          riskNotes: [],
          thinkMs: durationMs,
        };
        return {
          ...item,
          messages: [...item.messages, newMsg],
          isAnalyzing: false,
          thinkingStartedAt: null,
          lastThinkMs: durationMs,
          history: res.history || item.history,
          historyStack: res.history_stack || item.historyStack,
          meta: res.meta || item.meta,
          dataProfile: res.profile || item.dataProfile,
        };
      }));
    } catch (e) {
      const durationMs = Date.now() - startedAt;
      const isAbort = e?.name === 'AbortError';
      setSessions(prev => prev.map(item => item.id === sessionId ? { ...item, isAnalyzing: false, thinkingStartedAt: null, lastThinkMs: durationMs, messages: [...item.messages, { role: 'ai', content: isAbort ? "⏹ 已停止" : `❌ 分析失败：${e.message}`, thinkMs: durationMs }] } : item));
    } finally {
      delete abortRef.current[sessionId];
    }
  };

  useEffect(() => {
    if (!activeSessionId) return;
    (async () => {
      try {
        const response = await fetch(`${API_BASE}/session_state?session_id=${encodeURIComponent(activeSessionId)}`);
        if (!response.ok) return;
        const res = await response.json();
        setSessions(prev => prev.map(s => s.id === activeSessionId ? ({
          ...s,
          data: res.preview || [],
          dataMeta: { rows: res.rows || 0, cols: res.cols || 0, filename: res.filename || "未导入数据" },
          history: res.history || { cursor: 0, total: 0, can_undo: false, can_redo: false },
          historyStack: res.history_stack || [],
          meta: res.meta || null,
          dataProfile: res.profile || null,
        }) : s));
      } catch (e) {
        // 后端未启动/网络错误时忽略
      }
    })();
  }, [activeSessionId]);

  const toggleSuggestedAction = (sessionId, msgIdx, actionIdx) => {
    setSessions(prev => prev.map(s => {
      if (s.id !== sessionId) return s;
      const msgs = (s.messages || []).map((m, i) => {
        if (i !== msgIdx) return m;
        if (!m.actions) return m;
        const nextActions = m.actions.map((a, j) => j === actionIdx ? { ...a, checked: !(a.checked !== false) } : a);
        return { ...m, actions: nextActions };
      });
      return { ...s, messages: msgs };
    }));
  };

  const handleApplySuggestedActions = async (sessionId, msgIdx) => {
    const s = sessions.find(x => x.id === sessionId);
    const msg = s?.messages?.[msgIdx];
    const selected = (msg?.actions || []).filter(a => a.checked !== false).map(({ checked, ...rest }) => rest);
    if (!selected.length) { alert("请至少选择一个操作"); return; }

    // 标记应用中
    setSessions(prev => prev.map(item => {
      if (item.id !== sessionId) return item;
      return {
        ...item,
        messages: item.messages.map((m, i) => i === msgIdx ? { ...m, isApplyingActions: true } : m)
      };
    }));

    try {
      const runMode = s?.runMode || 'agent_single'; // ask | agent_single | agent_multi
      const ms = (s?.modelPrefs?.agent_single) || { provider: 'deepseekA', model: 'deepseek-reasoner' };
      const roles = (s?.modelPrefs?.agent_multi) || {
        planner: { provider: 'deepseekA', model: 'deepseek-reasoner' },
        executor: { provider: 'deepseekB', model: 'deepseek-reasoner' },
        verifier: { provider: 'deepseekC', model: 'deepseek-reasoner' },
      };

      const response = await fetch(`${API_BASE}/apply_actions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          actions: selected,
          // 让后端在“清洗建议 + 后续分析”的场景下自动续跑 Agent（更像真 agent）
          apiKeys: getEffectiveApiKeys(),
          mode: runMode,
          modelSelection: ms,
          agentRoles: roles,
          autoContinue: true,
        })
      });
      const res = await response.json();
      if (!response.ok) throw new Error(res.detail || "应用失败");

      setSessions(prev => prev.map(item => {
        if (item.id !== sessionId) return item;
        const updatedMessages = item.messages.map((m, i) => i === msgIdx ? { ...m, isApplyingActions: false, actionsApplied: true } : m);
        const appliedMsg = { role: 'ai', content: res.message || "已应用操作。", code: res.generated_code, result: null, image: null, log: '' };
        const follow = res.agent_followup;
        const followMsg = (follow && follow.reply) ? { role: 'ai', content: follow.reply, code: follow.generated_code || null, result: follow.execution_result || null, image: follow.image || null, tables: follow.tables || [], images: follow.images || [], log: follow.process_log || '' } : null;
        return {
          ...item,
          messages: [...updatedMessages, appliedMsg, ...(followMsg ? [followMsg] : [])],
          data: res.new_data_preview || item.data,
          dataMeta: { ...item.dataMeta, rows: res.rows ?? item.dataMeta.rows, cols: res.cols ?? item.dataMeta.cols },
          history: res.history || item.history,
          historyStack: res.history_stack || item.historyStack,
          meta: res.meta || item.meta,
          dataProfile: res.profile || item.dataProfile,
        };
      }));
    } catch (e) {
      setSessions(prev => prev.map(item => {
        if (item.id !== sessionId) return item;
        return {
          ...item,
          messages: item.messages.map((m, i) => i === msgIdx ? { ...m, isApplyingActions: false } : m)
        };
      }));
      alert("应用失败：" + e.message);
    }
  };

  const handleUndo = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE}/undo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      const res = await response.json();
      if (!response.ok) throw new Error(res.detail || "撤销失败");
      if (res.new_data_preview) {
        updateSession(sessionId, {
          data: res.new_data_preview,
          dataMeta: { ...activeSession.dataMeta, rows: res.rows ?? activeSession.dataMeta.rows, cols: res.cols ?? activeSession.dataMeta.cols },
          history: res.history || activeSession.history,
          historyStack: res.history_stack || activeSession.historyStack,
          meta: res.meta || activeSession.meta,
          dataProfile: res.profile || activeSession.dataProfile,
        });
      } else {
        updateSession(sessionId, { history: res.history || activeSession.history });
      }
    } catch (e) {
      alert("撤销失败：" + e.message);
    }
  };

  const handleRedo = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE}/redo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      const res = await response.json();
      if (!response.ok) throw new Error(res.detail || "重做失败");
      if (res.new_data_preview) {
        updateSession(sessionId, {
          data: res.new_data_preview,
          dataMeta: { ...activeSession.dataMeta, rows: res.rows ?? activeSession.dataMeta.rows, cols: res.cols ?? activeSession.dataMeta.cols },
          history: res.history || activeSession.history,
          historyStack: res.history_stack || activeSession.historyStack,
          meta: res.meta || activeSession.meta,
          dataProfile: res.profile || activeSession.dataProfile,
        });
      } else {
        updateSession(sessionId, { history: res.history || activeSession.history });
      }
    } catch (e) {
      alert("重做失败：" + e.message);
    }
  };

  const addSession = () => {
    const newSession = createNewSession(sessions.length);
    setSessions([...sessions, newSession]);
    setActiveSessionId(newSession.id);
  };

  const removeSession = async (e, id) => {
    e.stopPropagation();
    if (sessions.length === 1) { alert("至少保留一个任务窗口"); return; }
    
    // 清理该任务的 out 目录
    try {
      const formData = new FormData();
      formData.append("session_id", id);
      await fetch(`${API_BASE}/cleanup_out`, { method: 'POST', body: formData });
    } catch (err) {
      console.warn("清理 out 目录失败:", err);
    }
    
    const newSessions = sessions.filter(s => s.id !== id);
    setSessions(newSessions);
    if (activeSessionId === id) setActiveSessionId(newSessions[0].id);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => alert("已复制到剪贴板！"));
  };

  const handleDownloadReport = async () => {
      if (!activeSession.reportContent) return;
      const response = await fetch(`${API_BASE}/download_report`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ content: activeSession.reportContent, filename: `Radarm_Report_${activeSession.name}` })
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = `Radarm_Report_${activeSession.name}.md`;
      a.click();
  };

  const handlePrintReport = () => {
      const printWindow = window.open('', '_blank');
      printWindow.document.write(`<html><head><title>Radarm Report</title><style>body{font-family:sans-serif;line-height:1.6;padding:40px;max-width:800px;margin:0 auto;}</style></head><body><h1>Radarm 数据分析报告</h1>${activeSession.reportContent.replace(/\n/g, '<br/>')}</body></html>`);
      printWindow.document.close();
      printWindow.print();
  };

      // ✅ 导入后自动预检：理解数据 + 给出清洗(Action卡片)与分析建议
      const runOnboardSuggest = async (sessionId) => {
    const sNow = (sessions || []).find(x => x.id === sessionId) || activeSession;
    const prefs = sNow?.modelPrefs || {};
    const ms = prefs.agent_single || { provider: 'deepseekA', model: 'deepseek-reasoner' };
    const preferProvider = String(ms.provider || 'deepseekA');
    const preferModel = String(ms.model || getDefaultModelFor(preferProvider, 'agent_single'));

    const startedAt = Date.now();
    // 标记“思考中”
    setSessions(prev => prev.map(item => item.id === sessionId ? { ...item, isAnalyzing: true, thinkingStartedAt: startedAt } : item));

    try {
      const controller = new AbortController();
      abortRef.current[sessionId] = controller;
      const resp = await fetch(`${API_BASE}/onboard_suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          apiKeys: getEffectiveApiKeys(),
          provider: preferProvider,
          model: preferModel,
        }),
        signal: controller.signal
      });
      const ob = await resp.json();
      const durationMs = Date.now() - startedAt;

      const suggestedActions = (ob.suggested_actions || []).map(a => ({ ...a, checked: true }));
      const images = Array.isArray(ob.images) ? ob.images : ((ob.charts || []).map(c => c?.path).filter(Boolean));
      const tables = Array.isArray(ob.tables) ? ob.tables : [];

      const newMsg = {
        role: 'ai',
            content: ob.reply || "✅ 已完成导入后预检。",
        code: null,
        result: null,
        image: ob.image || (images.length ? images[0] : null),
        images,
        tables,
        log: ob.process_log || '',
        actions: suggestedActions.length ? suggestedActions : null,
        needsConfirmation: !!ob.needs_confirmation,
        riskNotes: ob.risk_notes || [],
        actionsApplied: false,
        isApplyingActions: false,
        thinkMs: durationMs,
      };

      setSessions(prev => prev.map(item => item.id === sessionId ? {
        ...item,
        messages: [...(item.messages || []), newMsg],
        isAnalyzing: false,
        thinkingStartedAt: null,
        lastThinkMs: durationMs,
      } : item));
    } catch (e) {
      const durationMs = Date.now() - startedAt;
      const isAbort = e?.name === 'AbortError';
      setSessions(prev => prev.map(item => item.id === sessionId ? {
        ...item,
        isAnalyzing: false,
        thinkingStartedAt: null,
        lastThinkMs: durationMs,
            messages: [...(item.messages || []), { role: 'ai', content: isAbort ? "⏹ 已停止（导入后预检）" : `⚠️ 导入后预检失败：${e.message || '未知错误'}`, thinkMs: durationMs }]
      } : item));
    } finally {
      delete abortRef.current[sessionId];
    }
  };

  const handleFileUpload = async (event, sessionId) => {
    const file = event.target.files?.[0];
    if (!file) return;
    updateSession(sessionId, { isUploading: true });
    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", sessionId);
    try {
      const response = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error("上传失败");
      const resData = await response.json();
      updateSession(sessionId, {
        data: resData.preview,
        dataMeta: { rows: resData.rows, cols: resData.cols, filename: resData.filename },
        history: resData.history || { cursor: 0, total: 0, can_undo: false, can_redo: false },
        historyStack: resData.history_stack || [],
        meta: resData.meta || null,
        dataProfile: resData.profile || null,
        messages: [...sessions.find(s=>s.id===sessionId).messages, { role: 'ai', content: `成功导入 **${resData.filename}**` }]
      });
      runOnboardSuggest(sessionId);
    } catch (error) { alert("上传出错：" + error.message); } 
    finally { updateSession(sessionId, { isUploading: false }); event.target.value = ''; }
  };

  const handleAskImageUpload = async (event, sessionId) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const s = sessions.find(x => x.id === sessionId);
    if (!s) return;
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("session_id", sessionId);
      const response = await fetch(`${API_BASE}/upload_chat_image`, { method: 'POST', body: formData });
      const res = await response.json();
      if (!response.ok) throw new Error(res.detail || '上传失败');
      updateSession(sessionId, { askImages: [...(s.askImages || []), res.path] });
    } catch (e) {
      alert("图片上传失败：" + e.message);
    } finally {
      event.target.value = '';
    }
  };

  const handleConnectDB = async (sessionId) => {
    updateSession(sessionId, { isUploading: true }); 
    try {
      const response = await fetch(`${API_BASE}/connect_db`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...dbConfig, session_id: sessionId })
      });
      if (!response.ok) throw new Error("连接失败");
      const resData = await response.json();
      updateSession(sessionId, {
        data: resData.preview,
        dataMeta: { rows: resData.rows, cols: resData.cols, filename: resData.filename },
        history: resData.history || { cursor: 0, total: 0, can_undo: false, can_redo: false },
        historyStack: resData.history_stack || [],
        meta: resData.meta || null,
        dataProfile: resData.profile || null,
        messages: [...sessions.find(s=>s.id===sessionId).messages, { role: 'ai', content: `成功连接数据库 **${resData.filename}**` }]
      });
      runOnboardSuggest(sessionId);
      setShowDbModal(false);
    } catch (error) { alert("数据库连接错误: " + error.message); } 
    finally { updateSession(sessionId, { isUploading: false }); }
  };

  const handleSendMessage = async (sessionId) => {
    const s = sessions.find(s => s.id === sessionId);
    if (!s.inputValue.trim()) return;

    const userMsg = s.inputValue;
    const imgPathsForSend = Array.isArray(s.askImages) ? [...s.askImages] : [];
    const startedAt = Date.now();
    setMentionOpen(false);

    const runMode = s.runMode || 'ask'; // ask | agent_single | agent_multi
    const prefs = s.modelPrefs || {};

    // Ask 模式允许“无数据”普通问答；Agent 模式需要先导入数据
    if (runMode !== 'ask' && (!s.data || s.data.length === 0)) { alert("请先导入数据"); return; }

    const pickProviderType = (pid) => pid && String(pid).startsWith('deepseek') ? 'deepseek' : String(pid || '');
    const getModelOptionsForProvider = (pid) => {
      const t = pickProviderType(pid);
      return MODEL_OPTIONS[t] || [];
    };

    const requiredProviders = (() => {
      if (runMode === 'agent_multi') {
        const r = prefs.agent_multi || {};
        const ps = [r.planner?.provider, r.executor?.provider, r.verifier?.provider].filter(Boolean);
        return Array.from(new Set(ps.map(String)));
      }
      const ms = (runMode === 'agent_single' ? prefs.agent_single : prefs.ask) || {};
      return ms.provider ? [String(ms.provider)] : [];
    })();

    const effKeys = getEffectiveApiKeys();
    const missingProviders = requiredProviders.filter(p => !effKeys?.[p]);
    if (requiredProviders.length === 0) {
      alert("请先选择模型/Provider"); setShowSettings(true); return;
    }
    if (missingProviders.length === requiredProviders.length) {
      const names = missingProviders.map(p => PROVIDER_LABELS[p] || p).join(', ');
      alert(`请配置 API Key（当前模式需要：${names}）`);
      setShowSettings(true);
      return;
    }
    if (runMode === 'agent_multi' && missingProviders.length > 0) {
      // 允许少 key 运行（后端会自动补位），但给出提醒
      alert(`提示：当前多专家缺少 Key：${missingProviders.join(', ')}。系统会自动用已配置的 Key 兜底，效果可能下降。`);
    }

    const payload = (() => {
      if (runMode === 'agent_multi') {
        const r = prefs.agent_multi || {};
        return {
          session_id: sessionId,
          message: userMsg,
          apiKeys: effKeys,
          mode: 'agent_multi',
          agentRoles: {
            planner: { provider: r.planner?.provider || 'deepseekA', model: r.planner?.model || 'deepseek-reasoner' },
            executor: { provider: r.executor?.provider || 'deepseekB', model: r.executor?.model || 'deepseek-reasoner' },
            verifier: { provider: r.verifier?.provider || 'deepseekC', model: r.verifier?.model || 'deepseek-reasoner' },
          },
          modelSelection: {},
          webSearch: !!s.askWebSearch,
          imagePaths: imgPathsForSend,
          visionEnabled: s.visionEnabled !== false,
          visionProvider: String(s.visionProvider || 'auto'),
          visionModel: (s.visionModel || '').trim() ? String((s.visionModel || '').trim()) : null,
        };
      }
      const ms = (runMode === 'agent_single' ? prefs.agent_single : prefs.ask) || {};
      const provider = String(ms.provider || 'deepseekA');
      const model = String(ms.model || (getModelOptionsForProvider(provider)[0] || ''));
      return {
        session_id: sessionId,
        message: userMsg,
        apiKeys: effKeys,
        mode: runMode === 'agent_single' ? 'agent_single' : 'ask',
        modelSelection: { provider, model },
        agentRoles: {},
        webSearch: !!s.askWebSearch,
        imagePaths: imgPathsForSend,
        visionEnabled: s.visionEnabled !== false,
        visionProvider: String(s.visionProvider || 'auto'),
        visionModel: (s.visionModel || '').trim() ? String((s.visionModel || '').trim()) : null,
      };
    })();

    // 发送后：把图片作为“用户消息附件”写入聊天流，并清空输入区附件（避免用户误以为没发出去）
    updateSession(sessionId, {
      messages: [...s.messages, { role: 'user', content: userMsg, images: imgPathsForSend }],
      inputValue: '',
      askImages: [],
      isAnalyzing: true,
      thinkingStartedAt: startedAt
    });

    try {
      const controller = new AbortController();
      abortRef.current[sessionId] = controller;
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      
      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorData.message || errorDetail;
        } catch (e) {
          try {
            const errorText = await response.text();
            if (errorText) errorDetail = errorText.substring(0, 200);
          } catch (e2) {
            // 忽略解析错误
          }
        }
        throw new Error(errorDetail);
      }
      
      const resData = await response.json();
      const durationMs = Date.now() - startedAt;
      
      setSessions(prev => prev.map(item => {
          if (item.id !== sessionId) return item;
          const suggestedActions = (resData.suggested_actions || []).map(a => ({ ...a, checked: true }));
          const images = Array.isArray(resData.images) ? resData.images : ((resData.charts || []).map(c => c?.path).filter(Boolean));
          const tables = Array.isArray(resData.tables) ? resData.tables : [];
          const newMsg = {
              role: 'ai',
              content: resData.reply,
              code: resData.generated_code,
              result: resData.execution_result,
              image: resData.image || (images.length ? images[0] : null),
              images,
              tables,
              log: resData.process_log,
              actions: suggestedActions.length ? suggestedActions : null,
              needsConfirmation: !!resData.needs_confirmation,
              riskNotes: resData.risk_notes || [],
              actionsApplied: false,
              isApplyingActions: false,
              thinkMs: durationMs,
          };
          return {
              ...item, messages: [...item.messages, newMsg], isAnalyzing: false, thinkingStartedAt: null, lastThinkMs: durationMs,
              data: resData.data_changed ? resData.new_data_preview : item.data,
              dataMeta: resData.data_changed ? { ...item.dataMeta, rows: resData.rows, cols: resData.cols } : item.dataMeta,
              history: resData.history || item.history,
              historyStack: resData.history_stack || item.historyStack,
              meta: resData.meta || item.meta,
              dataProfile: resData.profile || item.dataProfile,
          };
      }));
    } catch (error) {
       const durationMs = Date.now() - startedAt;
       const isAbort = error?.name === 'AbortError';
       const errorMsg = isAbort ? "⏹ 已停止" : (error?.message || "连接失败");
       setSessions(prev => prev.map(item => {
         if (item.id !== sessionId) return item;
         return {
           ...item,
           isAnalyzing: false,
           thinkingStartedAt: null,
           lastThinkMs: durationMs,
           messages: [...item.messages, { role: 'ai', content: `❌ ${errorMsg}`, thinkMs: durationMs }]
         };
       }));
    } finally {
      delete abortRef.current[sessionId];
    }
  };

  const handleStopThinking = (sessionId) => {
    const ctrl = abortRef.current?.[sessionId];
    if (ctrl) ctrl.abort();
  };

  const updateMentionState = (value, cursorPos, cols) => {
    const v = String(value || '');
    const cursor = typeof cursorPos === 'number' ? cursorPos : v.length;
    const before = v.slice(0, cursor);
    const atIdx = before.lastIndexOf('@');
    if (atIdx < 0) { setMentionOpen(false); setMentionQuery(''); setMentionStart(-1); return; }
    const prevChar = atIdx === 0 ? ' ' : before[atIdx - 1];
    if (prevChar && !/\s|[，,。;；:：]/.test(prevChar)) { setMentionOpen(false); return; }
    const after = before.slice(atIdx + 1);
    if (/[\s，,。;；:：]/.test(after)) { setMentionOpen(false); return; }
    setMentionStart(atIdx);
    setMentionQuery(after);
    setMentionOpen(true);
  };

  const openMentionPicker = () => {
    const s = activeSession;
    const v = String(s?.inputValue || '');
    const inputEl = chatInputRef.current;
    const cursor = inputEl?.selectionStart ?? v.length;
    setMentionStart(cursor);
    setMentionQuery('');
    setMentionOpen(true);
    setTimeout(() => {
      try { chatInputRef.current?.focus(); } catch {}
    }, 0);
  };

  const insertMention = (col) => {
    if (!mentionOpen || mentionStart < 0) return;
    const s = activeSession;
    const v = String(s.inputValue || '');
    const inputEl = chatInputRef.current;
    const cursor = inputEl?.selectionStart ?? v.length;
    const before = v.slice(0, mentionStart);
    const after = v.slice(cursor);
    const next = `${before}@${col} ${after}`;
    updateSession(activeSession.id, { inputValue: next });
    setMentionOpen(false);
    setMentionQuery('');
    setMentionStart(-1);
    // restore cursor after inserted mention
    setTimeout(() => {
      try {
        const pos = (before.length + 1 + String(col).length + 1);
        chatInputRef.current?.focus();
        chatInputRef.current?.setSelectionRange(pos, pos);
      } catch {}
    }, 0);
  };

  const refreshReportList = async (sessionId) => {
    try {
      const resp = await fetch(`${API_BASE}/report_list?session_id=${encodeURIComponent(sessionId)}`);
      const res = await resp.json();
      if (!resp.ok) throw new Error(res.detail || '获取报告列表失败');
      updateSession(sessionId, { reports: Array.isArray(res.reports) ? res.reports : [] });
    } catch (e) {
      // 静默失败：不打断主流程
    }
  };

  const openReport = async (sessionId, reportId) => {
    try {
      const resp = await fetch(`${API_BASE}/report_get?session_id=${encodeURIComponent(sessionId)}&report_id=${encodeURIComponent(reportId)}`);
      const res = await resp.json();
      if (!resp.ok) throw new Error(res.detail || '获取报告失败');
      const manifestTables = res?.manifest?.artifacts?.tables;
      updateSession(sessionId, {
        activeReportId: res.report_id || reportId,
        reportContent: res.report || '',
        reportLog: res.process_log || '',
        reportCharts: res.charts || [],
        reportTables: Array.isArray(manifestTables) ? manifestTables : [],
      });
    } catch (e) {
      alert(`打开报告失败：${e.message}`);
    }
  };

  const handleDownloadReportBundle = async (sessionId) => {
    const s = sessions.find(x => x.id === sessionId);
    if (!s?.activeReportId) return alert("请先选择一份报告");
    try {
      const resp = await fetch(`${API_BASE}/download_report_bundle?session_id=${encodeURIComponent(sessionId)}&report_id=${encodeURIComponent(s.activeReportId)}`);
      if (!resp.ok) throw new Error("下载失败");
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Radarm_Report_${s.activeReportId}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      alert(`导出 ZIP 失败：${e.message}`);
    }
  };

  const handleCancelReport = async (sessionId) => {
    const s = sessions.find(s => s.id === sessionId);
    if (!s || !s.isGeneratingReport) return;
    
    try {
      // 如果知道具体的 report_id，取消特定报告；否则取消所有正在生成的报告
      const reportId = s.generatingReportId;
      const response = await fetch(`${API_BASE}/report_cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, report_id: reportId || null })
      });
      const res = await response.json();
      if (res.cancelled) {
        updateSession(sessionId, { 
          isGeneratingReport: false, 
          generatingReportId: null,
          reportContent: '⚠️ 报告生成已取消',
          reportLog: res.message || '已取消'
        });
      }
    } catch (e) {
      console.error('取消报告失败:', e);
      // 即使请求失败，也更新前端状态
      updateSession(sessionId, { 
        isGeneratingReport: false, 
        generatingReportId: null
      });
    }
  };

  const handleGenerateReport = async (sessionId, opts = {}) => {
    const s = sessions.find(s => s.id === sessionId);
    if (!s) return;
    if (!s.data || s.data.length === 0) { alert("请导入数据"); return; }

    const draft = s.reportDraft || {};
    const stages = (draft.stages && typeof draft.stages === 'object') ? draft.stages : {};

    const requiredProviders = Array.from(new Set(
      ['planner','analyst','writer','reviewer']
        .map(k => stages?.[k]?.provider)
        .filter(Boolean)
        .map(String)
    ));
    const effKeys = getEffectiveApiKeys();
    const missingProviders = requiredProviders.filter(p => !effKeys?.[p]);
    if (missingProviders.length > 0) {
      const names = missingProviders.map(p => PROVIDER_LABELS[p] || p).join(', ');
      alert(`请先配置报告工作流所需 API Key：${names}（或在报告页把该阶段 provider 切换到已配置的服务）`);
      setShowSettings(true);
      return;
    }

    const saveAsNew = opts?.saveAsNew !== false;
    const overwrite = !!opts?.overwrite;
    const reportId = (!saveAsNew || overwrite) ? (s.activeReportId || null) : null;

    const payload = {
      session_id: sessionId,
      apiKeys: effKeys,
      userRequest: String(draft.userRequest || ''),
      selectedColumns: Array.isArray(draft.selectedColumns) ? draft.selectedColumns : [],
      sampleRows: (draft.sampleRows && String(draft.sampleRows).trim()) ? Number(draft.sampleRows) : null,
      reportStages: stages,
      saveAsNew: saveAsNew && !overwrite,
      reportId: reportId,
    };

    // 设置生成状态（先设置一个临时 report_id，实际 report_id 由后端返回）
    const tempReportId = reportId || `temp_${Date.now()}`;
    updateSession(sessionId, { 
      isGeneratingReport: true, 
      generatingReportId: tempReportId,
      reportContent: '', 
      reportLog: '', 
      reportCharts: [], 
      reportTables: [] 
    }); 
    
    try {
      const response = await fetch(`${API_BASE}/report`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const res = await response.json();
      if (!response.ok) throw new Error(res.detail || res.error || '生成失败');

      // 如果响应中包含 report_id，立即更新 generatingReportId（用于取消）
      if (res.report_id) {
        updateSession(sessionId, { generatingReportId: res.report_id });
      }

      if (res.cancelled) {
        updateSession(sessionId, { 
          reportContent: '⚠️ 报告生成已取消', 
          reportLog: res.process_log || '',
          isGeneratingReport: false,
          generatingReportId: null
        });
      } else if (res.error) {
          updateSession(sessionId, { 
            reportContent: `❌ 生成失败: ${res.error}`, 
            reportLog: res.process_log || '',
            isGeneratingReport: false,
            generatingReportId: null
          });
      } else {
        updateSession(sessionId, {
          activeReportId: res.report_id,
          reportContent: res.report,
          reportLog: res.process_log || '',
          reportCharts: res.charts || [],
          reportTables: res.tables || [],
          reports: Array.isArray(res.reports) ? res.reports : (s.reports || []),
          isGeneratingReport: false,
          generatingReportId: null,
        });
      }
    } catch (e) {
      updateSession(sessionId, { 
        reportContent: `❌ 错误: ${e.message}`,
        isGeneratingReport: false,
        generatingReportId: null
      });
    } finally {
      refreshReportList(sessionId);
    }
  };

  // 进入“报告”页时，自动同步后端报告列表（支持多份报告预览）
  useEffect(() => {
    const s = activeSession;
    if (!s?.id) return;
    if ((s.activeTab || 'data') !== 'report') return;
    refreshReportList(s.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId, activeSession?.activeTab]);

  const handleReset = async (sessionId) => {
    if (!window.confirm("重置任务？")) return;
    try {
      const response = await fetch(`${API_BASE}/reset`, { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId}) });
      const res = await response.json();
      updateSession(sessionId, {
        data: res.preview || [],
        dataMeta: { rows: res.rows || 0, cols: res.cols || 0, filename: res.filename || "未导入数据" },
        history: res.history || { cursor: 0, total: 0, can_undo: false, can_redo: false },
        historyStack: res.history_stack || [],
        meta: res.meta || null,
        messages: [{ role: 'ai', content: res.message || '已重置。' }],
        reports: [],
        activeReportId: null,
        reportContent: '',
        reportLog: '',
        reportCharts: [],
        reportTables: [],
        reportDraft: createNewSession(0).reportDraft,
      });
    } catch (e) {
    updateSession(sessionId, { data: [], messages: [{ role: 'ai', content: '已重置。' }], reports: [], activeReportId: null, reportContent: '', reportLog: '', reportCharts: [], reportTables: [], reportDraft: createNewSession(0).reportDraft });
    }
  };

  const getGalleryImages = (msgs) => (msgs || [])
    .flatMap(m => (m?.images && Array.isArray(m.images) ? m.images : (m?.image ? [m.image] : [])))
    .filter(Boolean);
  const getImageUrl = (imgPath) => {
    if (!imgPath) return null;
    // 如果是 base64（旧格式），直接返回
    if (imgPath.startsWith('data:') || imgPath.length > 100) return imgPath;
    // 新格式：返回 /out/... URL
    return `${API_BASE}/out/${imgPath}`;
  };
  const isLogOpen = (sessionId, idx) => !!logExpanded[`${sessionId}:${idx}`];
  const toggleLog = (sessionId, idx) => {
    const key = `${sessionId}:${idx}`;
    setLogExpanded(prev => ({...prev, [key]: !prev[key]}));
  };
  const formatProcessLog = (logText) => {
    if (!logText) return '';
    try {
      // 尝试解析 JSON
      const parsed = JSON.parse(logText);
      // 格式化并返回
      return JSON.stringify(parsed, null, 2);
    } catch (e) {
      // 不是 JSON，原样返回
      return logText;
    }
  };

  return (
    <div className="flex h-screen bg-zinc-50 text-zinc-900 font-sans overflow-hidden selection:bg-indigo-100 selection:text-indigo-900">
      {/* Sidebar */}
      <div className={`${isSidebarOpen ? 'w-[280px]' : 'w-0'} bg-white flex flex-col border-r border-zinc-200/60 text-zinc-600 shrink-0 transition-all duration-300 overflow-hidden relative z-20`}>
        <div className="p-5 flex items-center justify-between border-b border-zinc-200/60 whitespace-nowrap">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-zinc-900 text-white flex items-center justify-center shadow-sm">
              <Layers size={18}/>
        </div>
            <span className="font-bold text-lg tracking-tight text-zinc-900">Radarm</span>
          </div>
          <button onClick={() => setIsSidebarOpen(false)} className="text-zinc-400 hover:text-zinc-900 transition-colors"><PanelLeftClose size={20} /></button>
        </div>
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
          {sessions.map((s) => (
            <div key={s.id} onClick={() => setActiveSessionId(s.id)} className={`group relative flex items-center justify-between p-3 rounded-2xl cursor-pointer transition-all border ${activeSessionId === s.id ? 'bg-zinc-900 text-white border-zinc-900 shadow-xl shadow-zinc-200' : 'bg-white border-transparent hover:bg-zinc-50 text-zinc-600 hover:text-zinc-900'}`}>
              <div className="flex items-center gap-3 overflow-hidden">
                 <div className={`w-2 h-2 shrink-0 rounded-full transition-colors ${s.isAnalyzing ? 'bg-indigo-400 animate-pulse' : (activeSessionId === s.id ? 'bg-white/40' : 'bg-zinc-300')}`}></div>
                 <div className="flex flex-col min-w-0">
                   <span className="font-medium text-sm truncate">{s.name}</span>
                   <span className={`text-[10px] truncate ${activeSessionId === s.id ? 'text-zinc-400' : 'text-zinc-400'}`}>{s.dataMeta.filename}</span>
              </div>
              </div>
              <button onClick={(e) => removeSession(e, s.id)} className={`opacity-0 group-hover:opacity-100 p-1.5 rounded-lg transition-all ${activeSessionId === s.id ? 'hover:bg-white/20 text-white' : 'hover:bg-zinc-200 text-zinc-500'}`}><X size={14}/></button>
            </div>
          ))}
        </div>
        <div className="p-4 border-t border-zinc-200/60 whitespace-nowrap">
          <button onClick={addSession} className="w-full flex items-center justify-center gap-2 bg-zinc-900 text-white py-3 rounded-xl transition-all text-sm font-medium hover:bg-zinc-800 shadow-sm">
            <Plus size={16}/> 新建任务
          </button>
        </div>
      </div>

      {/* Main Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#FAFAFA] relative z-10">
        <header className="h-16 border-b border-zinc-200/60 flex items-center justify-between px-6 shrink-0 bg-[#FAFAFA]/80 backdrop-blur-sm sticky top-0 z-30">
          <div className="flex items-center gap-4">
             {!isSidebarOpen && <button onClick={() => setIsSidebarOpen(true)} className="text-zinc-400 hover:text-indigo-600 transition-colors mr-2"><PanelLeft size={24} /></button>}
             <input value={activeSession.name} onChange={(e) => updateSession(activeSession.id, { name: e.target.value })} className="font-bold text-lg text-zinc-900 bg-transparent border-none focus:ring-0 p-0 w-48 focus:bg-white/60 rounded-lg transition-all"/>
             <div className="h-6 w-px bg-zinc-200"></div>
             <div className="flex p-1 bg-zinc-200/50 rounded-2xl">
                {['data','visual','report'].map(t => (
                    <button key={t} onClick={() => updateSession(activeSession.id, { activeTab: t })} className={`relative px-4 py-2 rounded-xl text-sm font-medium transition-all ${activeSession.activeTab === t ? 'bg-white text-zinc-900 shadow-sm' : 'text-zinc-500 hover:text-zinc-700 hover:bg-white/40'}`}>{t==='data'?'数据':t==='visual'?'图表':'报告'}</button>
                ))}
             </div>
          </div>
          <div className="flex items-center gap-3">
             <div className="flex items-center gap-1 bg-white p-1 rounded-2xl border border-zinc-100 shadow-sm">
               <button onClick={() => setShowDbModal(true)} className="p-2.5 rounded-xl text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 transition-all" title="连接数据库"><Server size={18}/></button>
               <button onClick={() => handleUndo(activeSession.id)} disabled={!activeSession.history?.can_undo} className="p-2.5 rounded-xl text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 transition-all disabled:opacity-40" title={`撤销（${activeSession.history?.cursor || 0}/${activeSession.history?.total || 0}）`}><RotateCcw size={18}/></button>
               <button onClick={() => handleRedo(activeSession.id)} disabled={!activeSession.history?.can_redo} className="p-2.5 rounded-xl text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 transition-all disabled:opacity-40" title={`重做（${activeSession.history?.cursor || 0}/${activeSession.history?.total || 0}）`}><RotateCw size={18}/></button>
               <button onClick={() => { setToolDrawerOpen(true); setToolTab('history'); }} className="p-2.5 rounded-xl text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 transition-all" title="工具面板（列属性/历史/清洗）"><SlidersHorizontal size={18}/></button>
               <button onClick={() => handleReset(activeSession.id)} className="p-2.5 rounded-xl text-red-400 hover:text-red-600 hover:bg-red-50 transition-all" title="重置"><Trash2 size={18}/></button>
               <button onClick={() => setShowSettings(true)} className="p-2.5 rounded-xl text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 transition-all" title="设置"><Settings size={18}/></button>
             </div>
             <div className="h-6 w-px bg-zinc-200 mx-1"></div>
             <input type="file" ref={fileInputRef} onChange={(e) => handleFileUpload(e, activeSession.id)} className="hidden" accept=".csv,.xlsx,.json,.parquet"/>
             <button onClick={() => fileInputRef.current?.click()} disabled={activeSession.isUploading} className="bg-zinc-900 text-white px-5 py-2.5 rounded-xl text-sm font-medium hover:bg-zinc-800 flex items-center gap-2 shadow-lg shadow-zinc-200 transition-transform active:scale-95 disabled:opacity-60">{activeSession.isUploading ? <Loader2 size={16} className="animate-spin"/> : <Upload size={16}/>} 导入数据</button>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden px-6 pb-6 gap-6">
           <div className="flex-1 flex flex-col overflow-hidden min-w-0">
              <div className="bg-white rounded-3xl border border-zinc-200/60 shadow-sm flex-1 overflow-hidden relative">
                 <div className="absolute inset-0 overflow-auto">
                    {activeSession.activeTab === 'data' && (
                       activeSession.data.length > 0 ? (
                         <div className="flex flex-col h-full">
                            <div className="px-4 py-2 bg-yellow-50 text-yellow-700 text-xs border-b border-yellow-100 flex justify-between items-center shrink-0"><span>⚠️ 为保障性能，当前仅预览前 2000 行数据。完整数据共 {activeSession.dataMeta.rows} 行。</span><span className="font-mono opacity-75">{activeSession.dataMeta.cols} 列</span></div>
                            <div className="flex-1 overflow-auto w-full">
                                <table className="w-full text-left border-collapse text-sm">
                                <thead className="bg-slate-50 sticky top-0 z-10"><tr>{Object.keys(activeSession.data[0]).map(k=><th key={k} onClick={() => openColumnTools(k)} title="点击编辑列属性 / 菜单化清洗" className="px-4 py-2 font-semibold text-slate-600 border-b cursor-pointer hover:bg-slate-100">{k}</th>)}</tr></thead>
                                <tbody>{activeSession.data.map((r,i)=><tr key={i} className="hover:bg-slate-50 border-b border-slate-100">{Object.values(r).map((v,j)=><td key={j} className="px-4 py-2 text-slate-600 max-w-[200px] truncate">{typeof v==='object' && v!==null?JSON.stringify(v):v}</td>)}</tr>)}</tbody>
                                </table>
                            </div>
                         </div>
                       ) : <div className="h-full flex flex-col items-center justify-center text-slate-400"><Database size={40} className="mb-2 opacity-20"/><p>暂无数据</p></div>
                    )}
                    {activeSession.activeTab === 'visual' && (
                       getGalleryImages(activeSession.messages).length > 0 ? (
                         <div className="p-4 grid grid-cols-2 gap-4">{getGalleryImages(activeSession.messages).map((src, i) => {
                           const url = getImageUrl(src);
                           return url ? (<img key={i} alt="" src={url} className="w-full rounded border shadow-sm cursor-pointer hover:scale-[1.02]" onClick={()=>setPreviewImage(url)}/>) : null;
                         })}</div>
                       ) : <div className="h-full flex flex-col items-center justify-center text-slate-400"><BarChart2 size={40} className="mb-2 opacity-20"/><p>暂无图表</p></div>
                    )}
                    {activeSession.activeTab === 'report' && (
                       <div className="p-6 h-full bg-gradient-to-br from-slate-50 to-slate-100">
                          <div className="flex h-full gap-6">
                            {/* 左侧：生成设置 + 报告列表 */}
                            <div className="w-[360px] shrink-0 flex flex-col gap-4">
                              <div className="flex items-center justify-between bg-white rounded-lg shadow-sm px-4 py-3 border border-slate-200">
                                <div className="flex items-center gap-2">
                                  <FileText size={18} className="text-indigo-600"/>
                                  <div className="font-bold text-slate-800">数据分析报告</div>
                                </div>
                                <button onClick={() => refreshReportList(activeSession.id)} className="text-xs px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 hover:border-slate-300 transition-colors flex items-center gap-1">
                                  <RotateCcw size={12}/>
                                  刷新
                                </button>
                              </div>

                              <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-4 space-y-4">
                                <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
                                  <Zap size={14} className="text-indigo-600"/>
                                  <div className="text-sm font-bold text-slate-800">生成设置</div>
                                </div>

                                <div>
                                  <div className="text-xs font-semibold text-slate-700 mb-2 flex items-center gap-1">
                                    <AtSign size={12} className="text-indigo-500"/>
                                    分析需求（可选，可用 @列名）
                                  </div>
                                  <textarea
                                    value={activeSession.reportDraft?.userRequest || ''}
                                    onChange={(e) => updateSession(activeSession.id, { reportDraft: { ...(activeSession.reportDraft || {}), userRequest: e.target.value } })}
                                    className="w-full border border-slate-300 rounded-lg px-3 py-2 text-xs bg-slate-50 focus:bg-white focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all resize-none"
                                    placeholder="例如：重点分析 @身高 @体重 与结局变量的关系；并给出可执行建议。"
                                    rows={4}
                                  />
                                </div>

                                <div>
                                  <div className="text-xs font-semibold text-slate-700 mb-2 flex items-center gap-1">
                                    <Layers size={12} className="text-indigo-500"/>
                                    选取字段（可选，留空=全字段）
                                  </div>
                                  <input
                                    value={reportColSearch}
                                    onChange={(e) => setReportColSearch(e.target.value)}
                                    className="w-full border border-slate-300 rounded-lg px-3 py-2 text-xs bg-slate-50 focus:bg-white focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all"
                                    placeholder="搜索字段..."
                                  />

                                  {(() => {
                                    const cols = getColumnsList(activeSession);
                                    const selected = Array.isArray(activeSession.reportDraft?.selectedColumns) ? activeSession.reportDraft.selectedColumns : [];
                                    const q = (reportColSearch || '').trim().toLowerCase();
                                    const list = cols.filter(c => !q || String(c).toLowerCase().includes(q));
                                    const toggleCol = (c) => {
                                      const cur = selected;
                                      const exists = cur.includes(c);
                                      const next = exists ? cur.filter(x => x !== c) : [...cur, c];
                                      updateSession(activeSession.id, { reportDraft: { ...(activeSession.reportDraft || {}), selectedColumns: next } });
                                    };
                                    return (
                                      <div className="mt-2 border border-slate-200 rounded-lg bg-slate-50 max-h-40 overflow-y-auto shadow-inner">
                                        {list.length ? list.map(c => (
                                          <label key={c} className="flex items-center gap-2 px-3 py-1.5 text-xs text-slate-700 hover:bg-white cursor-pointer transition-colors border-b border-slate-100 last:border-b-0">
                                            <input type="checkbox" checked={selected.includes(c)} onChange={() => toggleCol(c)} className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500" />
                                            <span className="truncate flex-1">{c}</span>
                                          </label>
                                        )) : (
                                          <div className="px-3 py-3 text-xs text-slate-400 text-center">未找到匹配字段</div>
                                        )}
                                      </div>
                                    );
                                  })()}

                                  <div className="flex items-center justify-between mt-2">
                                    <div className="text-[10px] text-slate-500">
                                      已选 {(activeSession.reportDraft?.selectedColumns || []).length} / {getColumnsList(activeSession).length}
                                    </div>
                                    <div className="flex gap-2">
                                      <button
                                        onClick={() => updateSession(activeSession.id, { reportDraft: { ...(activeSession.reportDraft || {}), selectedColumns: getColumnsList(activeSession) } })}
                                        className="text-xs px-2 py-1 rounded border border-slate-200 bg-white hover:bg-slate-50"
                                      >
                                        全选
                                </button>
                                      <button
                                        onClick={() => updateSession(activeSession.id, { reportDraft: { ...(activeSession.reportDraft || {}), selectedColumns: [] } })}
                                        className="text-xs px-2 py-1 rounded border border-slate-200 bg-white hover:bg-slate-50"
                                      >
                                        清空
                                      </button>
                                    </div>
                                  </div>

                                  <div className="text-[10px] text-slate-500 mt-2 px-2 py-1 bg-slate-50 rounded border border-slate-200">
                                    💡 提示：不选=全字段；也可在"分析需求"里用 @列名 点名。
                                  </div>
                                </div>

                                <div>
                                  <div className="text-xs font-semibold text-slate-700 mb-2 flex items-center gap-1">
                                    <Database size={12} className="text-indigo-500"/>
                                    抽样行数（可选）
                                  </div>
                                  <input
                                    value={activeSession.reportDraft?.sampleRows || ''}
                                    onChange={(e) => updateSession(activeSession.id, { reportDraft: { ...(activeSession.reportDraft || {}), sampleRows: e.target.value } })}
                                    className="w-full border border-slate-300 rounded-lg px-3 py-2 text-xs bg-slate-50 focus:bg-white focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all"
                                    placeholder="留空=全量（建议≤10000）"
                                    type="number"
                                  />
                                </div>

                                <details className="bg-slate-50 border border-slate-200 rounded-lg p-3 shadow-sm">
                                  <summary className="text-xs font-semibold text-slate-700 cursor-pointer select-none flex items-center gap-2 hover:text-indigo-600 transition-colors">
                                    <Settings size={14} className="text-indigo-500"/>
                                    工作流模型（可选）
                                  </summary>
                                  <div className="mt-2 space-y-2">
                                    {(['planner','analyst','writer','reviewer']).map(stage => {
                                      const label = stage === 'planner' ? '规划' : stage === 'analyst' ? '洞察' : stage === 'writer' ? '成文' : '审校';
                                      const cur = (activeSession.reportDraft?.stages?.[stage]) || { provider: 'deepseekA', model: 'deepseek-reasoner' };
                                      const p = cur.provider || 'deepseekA';
                                      const opts = getProviderModelOptions(p);
                                      const m = (cur.model && opts.includes(cur.model)) ? cur.model : (getDefaultModelFor(p, 'agent_single') || opts[0] || '');
                                      return (
                                        <div key={stage} className="grid grid-cols-3 gap-2 items-center">
                                          <div className="text-[11px] font-bold text-slate-600">{label}</div>
                                          <select
                                            value={p}
                                            onChange={(e) => {
                                              const np = e.target.value;
                                              const nextModel = (getProviderModelOptions(np).includes(m) ? m : (getDefaultModelFor(np, 'agent_single') || getProviderModelOptions(np)[0] || ''));
                                              updateSession(activeSession.id, {
                                                reportDraft: {
                                                  ...(activeSession.reportDraft || {}),
                                                  stages: {
                                                    ...((activeSession.reportDraft && activeSession.reportDraft.stages) || {}),
                                                    [stage]: { provider: np, model: nextModel }
                                                  }
                                                }
                                              });
                                            }}
                                            className="w-full border rounded px-2 py-1 text-xs bg-white"
                                          >
                                            {allProviderIds.map(pid => <option key={pid} value={pid}>{PROVIDER_LABELS[pid] || pid}</option>)}
                                          </select>
                                          <select
                                            value={m}
                                            onChange={(e) => updateSession(activeSession.id, {
                                              reportDraft: {
                                                ...(activeSession.reportDraft || {}),
                                                stages: {
                                                  ...((activeSession.reportDraft && activeSession.reportDraft.stages) || {}),
                                                  [stage]: { provider: p, model: e.target.value }
                                                }
                                              }
                                            })}
                                            className="w-full border rounded px-2 py-1 text-xs bg-white"
                                          >
                                            {opts.map(mm => <option key={mm} value={mm}>{mm}</option>)}
                                          </select>
                                        </div>
                                      );
                                    })}
                                    <div className="text-[10px] text-slate-500">
                                      提示：所选 provider 必须在设置中配置对应 API Key；否则报告会失败。
                                    </div>
                                  </div>
                                </details>

                                <button
                                  onClick={() => handleGenerateReport(activeSession.id, { saveAsNew: true })}
                                  disabled={activeSession.isGeneratingReport}
                                  className="w-full bg-gradient-to-r from-indigo-600 to-indigo-700 text-white px-4 py-3 rounded-lg text-sm font-semibold hover:from-indigo-700 hover:to-indigo-800 disabled:opacity-50 flex items-center justify-center gap-2 shadow-md hover:shadow-lg transition-all"
                                >
                                  {activeSession.isGeneratingReport ? <Loader2 size={16} className="animate-spin"/> : <Zap size={16}/>}
                                  生成新报告
                                </button>

                                <button
                                  onClick={() => handleGenerateReport(activeSession.id, { overwrite: true, saveAsNew: false })}
                                  disabled={activeSession.isGeneratingReport || !activeSession.activeReportId}
                                  className="w-full bg-white text-slate-700 px-4 py-2.5 rounded-lg text-xs font-medium border-2 border-slate-300 hover:bg-slate-50 hover:border-slate-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                                  title="覆盖当前选中的报告（如未选择报告则不可用）"
                                >
                                  覆盖当前报告
                                </button>
                              </div>

                              <div className="flex-1 bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden flex flex-col">
                                <div className="px-4 py-3 text-sm font-bold text-slate-800 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white flex items-center gap-2">
                                  <FileText size={16} className="text-indigo-600"/>
                                  报告列表（{(activeSession.reports || []).length}）
                                </div>
                                <div className="flex-1 overflow-y-auto p-3 space-y-2">
                                  {(activeSession.reports || [])
                                    .slice()
                                    .sort((a, b) => (Number(b?.created_at || 0) - Number(a?.created_at || 0)))
                                    .map((r) => {
                                      const isActive = String(activeSession.activeReportId || '') === String(r.report_id || '');
                                      const title = r.title || '数据分析报告';
                                      const ts = r.created_at ? new Date(Number(r.created_at) * 1000).toLocaleString() : '';
                                      return (
                                        <button
                                          key={r.report_id}
                                          onClick={() => openReport(activeSession.id, r.report_id)}
                                          className={`w-full text-left p-3 rounded-lg border-2 transition-all ${
                                            isActive 
                                              ? 'border-indigo-500 bg-indigo-50 shadow-md' 
                                              : 'border-slate-200 hover:bg-slate-50 hover:border-slate-300 hover:shadow-sm'
                                          }`}
                                        >
                                          <div className="flex items-start gap-2">
                                            <FileText size={14} className={`mt-0.5 ${isActive ? 'text-indigo-600' : 'text-slate-400'}`}/>
                                            <div className="flex-1 min-w-0">
                                              <div className={`text-xs font-bold truncate ${isActive ? 'text-indigo-900' : 'text-slate-800'}`}>{title}</div>
                                              <div className="text-[10px] text-slate-500 mt-1 truncate">{ts}</div>
                                              {r.selected_columns && r.selected_columns.length > 0 && (
                                                <div className="text-[10px] text-slate-500 mt-1 truncate">字段：{r.selected_columns.slice(0, 6).join(', ')}{r.selected_columns.length > 6 ? '...' : ''}</div>
                                              )}
                                            </div>
                                          </div>
                                        </button>
                                      );
                                    })}
                                  {(!activeSession.reports || activeSession.reports.length === 0) && (
                                    <div className="text-xs text-slate-400 p-4 text-center bg-slate-50 rounded-lg border border-slate-200">
                                      <FileText size={24} className="mx-auto mb-2 opacity-30"/>
                                      <p>暂无报告</p>
                                      <p className="text-[10px] mt-1">点击上方"生成新报告"开始</p>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>

                            {/* 右侧：预览 */}
                            <div className="flex-1 bg-white border border-slate-200 rounded-xl shadow-lg overflow-hidden flex flex-col">
                              <div className="px-5 py-4 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <BarChart2 size={18} className="text-indigo-600"/>
                                  <div className="font-bold text-slate-800">报告预览</div>
                                </div>
                                <div className="flex items-center gap-2">
                                  <button onClick={() => copyToClipboard(activeSession.reportContent || '')} disabled={!activeSession.reportContent} className="flex items-center gap-1.5 text-xs bg-white hover:bg-slate-100 border border-slate-300 hover:border-slate-400 px-3 py-1.5 rounded-lg disabled:opacity-50 transition-all font-medium"><Copy size={13}/> 复制</button>
                                  <button onClick={handleDownloadReport} disabled={!activeSession.reportContent} className="flex items-center gap-1.5 text-xs bg-white hover:bg-slate-100 border border-slate-300 hover:border-slate-400 px-3 py-1.5 rounded-lg disabled:opacity-50 transition-all font-medium"><FileDown size={13}/> 导出MD</button>
                                  <button onClick={() => handleDownloadReportBundle(activeSession.id)} disabled={!activeSession.activeReportId} className="flex items-center gap-1.5 text-xs bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1.5 rounded-lg disabled:opacity-50 transition-all font-medium shadow-sm"><FileDown size={13}/> 导出ZIP</button>
                                  <button onClick={handlePrintReport} disabled={!activeSession.reportContent} className="flex items-center gap-1.5 text-xs bg-white hover:bg-slate-100 border border-slate-300 hover:border-slate-400 px-3 py-1.5 rounded-lg disabled:opacity-50 transition-all font-medium"><Printer size={13}/> 打印</button>
                                </div>
                              </div>
                              <div className="flex-1 overflow-y-auto p-6 bg-slate-50">
                                {activeSession.isGeneratingReport && (
                                  <div className="flex items-center justify-between gap-3 text-sm text-slate-600 mb-4 p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
                                    <div className="flex items-center gap-3">
                                      <Loader2 size={16} className="animate-spin text-indigo-600"/> 
                                      <span className="font-medium">正在生成报告（包含图表与图表数据）...</span>
                                    </div>
                                    <button
                                      onClick={() => handleCancelReport(activeSession.id)}
                                      className="flex items-center gap-1.5 text-xs bg-red-600 hover:bg-red-700 text-white px-3 py-1.5 rounded-lg transition-all font-medium shadow-sm"
                                    >
                                      <X size={12}/>
                                      停止生成
                                    </button>
                                  </div>
                                )}

                                {!activeSession.reportContent ? (
                                  <div className="h-full flex flex-col items-center justify-center text-slate-400">
                                    <div className="bg-white rounded-xl p-8 border-2 border-dashed border-slate-300">
                                      <FileText size={48} className="mx-auto mb-4 opacity-30"/>
                                      <p className="text-sm font-medium text-slate-600">请选择一份报告</p>
                                      <p className="text-xs text-slate-400 mt-1">或点击左侧"生成新报告"开始</p>
                                    </div>
                            </div>
                         ) : (
                                  <>
                                    {activeSession.reportCharts && Array.isArray(activeSession.reportCharts) && activeSession.reportCharts.length > 0 && (
                                      <div className="mb-6 bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
                                        <div className="flex items-center gap-2 text-sm font-bold text-slate-800 mb-3 pb-2 border-b border-slate-200">
                                          <BarChart2 size={16} className="text-indigo-600"/>
                                          📈 图表（{activeSession.reportCharts.length}）
                                </div>
                                        <div className="grid grid-cols-2 gap-3">
                                          {activeSession.reportCharts.slice(0, 12).map((c, i) => {
                                            const url = getImageUrl(c.path);
                                            return url ? (
                                              <div key={i} className="relative group rounded-lg overflow-hidden border border-slate-200 bg-white shadow-sm hover:shadow-md transition-all">
                                                <img
                                                  alt={c.name || ''}
                                                  src={url}
                                                  className="w-full cursor-pointer hover:scale-[1.02] transition-transform"
                                                  onClick={() => setPreviewImage(url)}
                                                  title={c.name || ''}
                                                />
                                                {c.name && (
                                                  <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[10px] px-2 py-1 truncate opacity-0 group-hover:opacity-100 transition-opacity">
                                                    {c.name}
                                                  </div>
                                                )}
                                              </div>
                                            ) : null;
                                          })}
                                        </div>
                                        {activeSession.reportCharts.length > 12 && (
                                          <div className="text-xs text-slate-500 mt-3 px-2 py-1 bg-slate-50 rounded border border-slate-200">
                                            💡 仅预览前 12 张图；导出 ZIP 可获取全部。
                                          </div>
                                        )}
                                      </div>
                                    )}

                                    {activeSession.reportTables && Array.isArray(activeSession.reportTables) && activeSession.reportTables.length > 0 && (
                                      <div className="mb-6 bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
                                        <div className="flex items-center gap-2 text-sm font-bold text-slate-800 mb-3 pb-2 border-b border-slate-200">
                                          <FileDown size={16} className="text-indigo-600"/>
                                          📎 图表数据（CSV）
                                        </div>
                                        <div className="space-y-2">
                                          {activeSession.reportTables.filter(t => !!t.csv_path).slice(0, 12).map((t, i) => {
                                            const url = getImageUrl(t.csv_path);
                                            return (
                                              <div key={i} className="flex items-center justify-between gap-3 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 hover:bg-white hover:border-slate-300 transition-all">
                                                <div className="flex items-center gap-2 flex-1 min-w-0">
                                                  <FileDown size={14} className="text-slate-400 shrink-0"/>
                                                  <div className="text-xs text-slate-700 truncate font-medium" title={t.name || ''}>{t.name || '表格'}</div>
                                                </div>
                                                {url ? (
                                                  <a href={url} className="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors font-medium shadow-sm" download>
                                                    下载CSV
                                                  </a>
                                                ) : (
                                                  <span className="text-[10px] text-slate-400">无CSV</span>
                                                )}
                                              </div>
                                            );
                                          })}
                                        </div>
                                        {activeSession.reportTables.filter(t => !!t.csv_path).length > 12 && (
                                          <div className="text-xs text-slate-500 mt-3 px-2 py-1 bg-slate-50 rounded border border-slate-200">
                                            💡 仅展示前 12 份 CSV；导出 ZIP 可获取全部。
                                          </div>
                                        )}
                                      </div>
                                    )}

                                {activeSession.reportLog && (
                                    <div className="mb-6 bg-indigo-50 border border-indigo-200 rounded-lg overflow-hidden shadow-sm">
                                        <div onClick={() => setReportLogExpanded(!reportLogExpanded)} className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-indigo-100 transition-colors">
                                          <span className="text-sm font-semibold text-indigo-800 flex items-center gap-2"><Users size={14}/> 查看工作流过程</span>
                                          {reportLogExpanded ? <ChevronUp size={14} className="text-indigo-600"/> : <ChevronDown size={14} className="text-indigo-600"/>}
                                        </div>
                                        {reportLogExpanded && (
                                          <div className="p-4 bg-white border-t border-indigo-100 max-h-64 overflow-y-auto">
                                            <pre className="text-[10px] text-slate-600 font-mono whitespace-pre-wrap">{formatProcessLog(activeSession.reportLog)}</pre>
                                    </div>
                                )}
                            </div>
                         )}

                                    <div className="bg-white rounded-lg p-6 border border-slate-200 shadow-sm">
                                      <div className="prose prose-sm max-w-none whitespace-pre-wrap text-slate-700 leading-relaxed">{activeSession.reportContent}</div>
                                    </div>
                                  </>
                         )}
                              </div>
                            </div>
                          </div>
                       </div>
                    )}
                 </div>
              </div>
           </div>

           <div className={`${isChatOpen ? 'w-[380px] overflow-visible' : 'w-12 overflow-hidden'} transition-all duration-300 ease-in-out bg-white rounded-3xl border border-zinc-200/60 shadow-xl shadow-zinc-200/50 flex flex-col shrink-0 relative z-20`}>
              {isChatOpen ? (
                <>
                  <div className="p-4 border-b border-zinc-100 flex items-center justify-between shrink-0 bg-white/90 backdrop-blur-md z-10">
                    <div className="flex items-center gap-2 text-zinc-800 font-bold">
                      <div className="w-8 h-8 rounded-full bg-indigo-50 flex items-center justify-center text-indigo-600"><MessageSquare size={16}/></div>
                      <span>助手</span>
                    </div>
                    <button onClick={() => setIsChatOpen(false)} className="p-2 rounded-full hover:bg-zinc-50 text-zinc-400 hover:text-zinc-600 transition-colors" title="收起聊天">
                      <ChevronRight size={18}/>
                    </button>
                  </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-zinc-50/30">
                 {activeSession.messages.map((msg, i) => (
                    <div key={i} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                       <div className={`max-w-[92%] p-4 rounded-2xl text-sm leading-relaxed shadow-sm whitespace-pre-wrap relative group ${msg.role==='user'?'bg-zinc-900 text-white rounded-br-sm':'bg-white text-zinc-700 border border-zinc-100 rounded-bl-sm'}`}>
                          {msg.content}
                          <button onClick={() => copyToClipboard(msg.content)} className={`absolute top-2 right-2 opacity-0 group-hover:opacity-100 p-1 rounded transition-opacity ${msg.role==='user'?'hover:bg-white/20 text-white/70':'hover:bg-zinc-100 text-zinc-400'}`}><Copy size={12}/></button>
                       </div>
                       {msg.role === 'ai' && typeof msg.thinkMs === 'number' && (
                         <div className="mt-1 text-[10px] text-zinc-400 px-1">{(msg.thinkMs / 1000).toFixed(1)}s</div>
                       )}
                       {msg.log && (
                         <div className="mt-2 w-[92%] bg-white border border-zinc-100 rounded-xl overflow-hidden shadow-sm">
                           <div onClick={() => toggleLog(activeSession.id, i)} className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-zinc-50 transition-colors">
                             <span className="text-xs font-semibold text-zinc-700 flex items-center gap-1"><Terminal size={12}/> 思考过程</span>
                             {isLogOpen(activeSession.id, i) ? <ChevronUp size={12} className="text-zinc-400"/> : <ChevronDown size={12} className="text-zinc-400"/>}
                           </div>
                           {isLogOpen(activeSession.id, i) && (
                             <div className="p-3 bg-zinc-900 border-t border-zinc-200 max-h-64 overflow-y-auto">
                               <pre className="text-[10px] text-zinc-400 font-mono whitespace-pre-wrap">{formatProcessLog(msg.log)}</pre>
                             </div>
                           )}
                         </div>
                       )}
                       {msg.images && Array.isArray(msg.images) && msg.images.length > 0 ? (
                         <div className="mt-2 flex flex-wrap gap-2">
                           {msg.images.map((p, idx) => {
                             const url = getImageUrl(p);
                             return url ? <img key={idx} alt="" src={url} className="h-20 w-auto rounded-lg border border-zinc-200 cursor-pointer hover:ring-2 hover:ring-indigo-500 transition-all" onClick={()=>setPreviewImage(url)}/> : null;
                           })}
                         </div>
                       ) : (
                         msg.image && (() => {
                           const url = getImageUrl(msg.image);
                           return url ? <img alt="" src={url} className="mt-2 max-w-[240px] rounded-lg border border-zinc-200 shadow-sm cursor-pointer hover:ring-2 hover:ring-indigo-500 transition-all" onClick={()=>setPreviewImage(url)}/> : null;
                         })()
                       )}
                       {msg.tables && Array.isArray(msg.tables) && msg.tables.length > 0 && (
                         <div className="mt-3 w-[92%] bg-zinc-50 border border-zinc-200 rounded-xl overflow-hidden">
                           <div className="px-3 py-2 border-b border-zinc-200 text-xs font-semibold text-zinc-700">表格结果</div>
                           <div className="p-3 space-y-3 max-h-80 overflow-y-auto">
                             {msg.tables.map((t, tidx) => (
                               <div key={tidx} className="bg-white border border-zinc-200 rounded-lg">
                                 <div className="px-3 py-2 border-b border-zinc-200 flex items-center justify-between">
                                   <span className="text-xs font-semibold text-zinc-700 truncate">{t.name || `表格${tidx+1}`}</span>
                                   <button onClick={() => copyToClipboard(t.markdown || '')} className="text-xs px-2 py-1 rounded border border-zinc-200 bg-white hover:bg-zinc-50">复制</button>
                                 </div>
                                 <div className="p-3 overflow-x-auto">
                                   <pre className="text-[11px] text-zinc-700 font-mono whitespace-pre">{t.markdown || ''}</pre>
                                 </div>
                               </div>
                             ))}
                           </div>
                         </div>
                       )}
                       {msg.code && (
                         <div className="mt-3 w-[92%] bg-zinc-900 rounded-xl border border-zinc-800 overflow-hidden shadow-md relative group">
                           <div className="bg-zinc-900/70 px-3 py-2 flex items-center justify-between border-b border-zinc-800">
                             <div className="flex items-center gap-2">
                               <Terminal size={12} className="text-emerald-400" />
                               <span className="text-xs text-zinc-300 font-mono">Python</span>
                             </div>
                             <button onClick={() => copyToClipboard(msg.code)} className="text-zinc-400 hover:text-white"><Copy size={12}/></button>
                           </div>
                           <div className="p-3 overflow-x-auto"><pre className="text-xs font-mono text-emerald-300 leading-tight">{msg.code}</pre></div>
                         </div>
                       )}
                       {msg.actions && msg.actions.length > 0 && (
                         <div className="mt-3 w-[92%] bg-indigo-50/50 border border-indigo-100 rounded-xl p-3">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-xs font-semibold text-indigo-900">建议操作</span>
                              {msg.actionsApplied && <span className="text-[11px] font-bold text-green-600">已应用</span>}
                    </div>
                            <div className="space-y-2">
                              {msg.actions.map((a, idx) => (
                                <label key={idx} className="flex items-start gap-2 text-xs text-zinc-700">
                                  <input
                                    type="checkbox"
                                    className="mt-0.5"
                                    checked={a.checked !== false}
                                    disabled={msg.actionsApplied || msg.isApplyingActions}
                                    onChange={() => toggleSuggestedAction(activeSession.id, i, idx)}
                                  />
                                  <span className="whitespace-pre-wrap">{formatSuggestedAction(a)}</span>
                                </label>
                              ))}
                            </div>
                            {msg.riskNotes && msg.riskNotes.length > 0 && (
                              <div className="mt-3 bg-yellow-50 border border-yellow-200 rounded p-2">
                                <div className="text-[11px] font-bold text-yellow-800 mb-1">风险提示</div>
                                <ul className="list-disc ml-4 text-[11px] text-yellow-800 space-y-1">
                                  {msg.riskNotes.map((r, ridx) => <li key={ridx}>{r}</li>)}
                                </ul>
                              </div>
                            )}
                            <button
                              onClick={() => handleApplySuggestedActions(activeSession.id, i)}
                              disabled={msg.actionsApplied || msg.isApplyingActions}
                              className={`mt-3 w-full text-xs py-2 rounded-lg ${msg.actionsApplied ? 'bg-zinc-200 text-zinc-500' : 'bg-indigo-600 text-white hover:bg-indigo-700'} disabled:opacity-60`}
                            >
                              {msg.isApplyingActions ? '应用中...' : (msg.actionsApplied ? '已应用' : '应用所选操作')}
                            </button>
                         </div>
                       )}
                    </div>
                 ))}
                    {activeSession.isAnalyzing && (
                      <div className="flex items-center gap-2 text-xs text-zinc-400 ml-2">
                        <Loader2 size={12} className="animate-spin"/>
                        Radarm 思考中...
                        {activeSession.thinkingStartedAt ? ` ${(Math.max(0, (thinkingTick - activeSession.thinkingStartedAt)) / 1000).toFixed(1)}s` : ""}
                      </div>
                    )}
                 <div ref={chatEndRef} />
              </div>

              <div className="p-4 bg-white border-t border-zinc-100 relative z-20">
                     {activeSession.askImages?.length > 0 && (
                       <div className="flex gap-2 mb-3 overflow-x-auto pb-1">
                         {activeSession.askImages.map((p, idx) => (
                           <div key={idx} className="relative group shrink-0">
                             <img src={getImageUrl(p)} className="w-12 h-12 rounded-lg object-cover border border-zinc-200" />
                             <button onClick={() => updateSession(activeSession.id, { askImages: activeSession.askImages.filter((_, i) => i !== idx) })} className="absolute -top-1 -right-1 bg-zinc-900 text-white rounded-full p-0.5 w-4 h-4 flex items-center justify-center text-[10px] opacity-0 group-hover:opacity-100 transition-opacity"><X size={8}/></button>
                          </div>
                         ))}
                    </div>
                  )}

                     <div className="relative bg-zinc-50 border border-zinc-200 rounded-2xl focus-within:ring-2 focus-within:ring-indigo-100 focus-within:border-indigo-300 transition-all shadow-inner">
                    <textarea
                      ref={chatInputRef}
                      value={activeSession.inputValue}
                      onChange={(e) => {
                        const v = e.target.value;
                        updateSession(activeSession.id, { inputValue: v });
                        updateMentionState(v, e.target.selectionStart, getColumnsList(activeSession));
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(activeSession.id); }
                            if (e.key === 'Escape') setMentionOpen(false);
                          }}
                          className="w-full bg-transparent border-none focus:ring-0 focus:outline-none p-3 text-sm text-zinc-800 placeholder-zinc-400 min-h-[44px] max-h-32 resize-none"
                          placeholder="输入指令或提问..."
                          rows={1}
                        />
                        <input type="file" ref={imageInputRef} onChange={(e) => handleAskImageUpload(e, activeSession.id)} className="hidden" accept="image/*"/>
                        <div className="flex items-center justify-between px-2 pb-2">
                           <div className="flex items-center gap-1">
                            <button
                                onClick={() => setShowModeModal(true)}
                                className="p-1.5 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-200/50 rounded-lg transition-colors"
                                title="模式与模型设置"
                              >
                                <Command size={16}/>
                            </button>
                              <button onClick={() => updateSession(activeSession.id, { askWebSearch: !activeSession.askWebSearch })} className={`p-1.5 rounded-lg transition-colors ${activeSession.askWebSearch ? 'text-blue-500 bg-blue-50' : 'text-zinc-400 hover:text-zinc-600 hover:bg-zinc-200/50'}`}><Globe size={16}/></button>
                              <button onClick={() => imageInputRef.current?.click()} className="p-1.5 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-200/50 rounded-lg transition-colors"><ImageIcon size={16}/></button>
                              <button onClick={openMentionPicker} className="p-1.5 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-200/50 rounded-lg transition-colors"><AtSign size={16}/></button>
           </div>

                      {activeSession.isAnalyzing ? (
                             <button onClick={() => handleStopThinking(activeSession.id)} className="w-8 h-8 flex items-center justify-center bg-zinc-200 text-zinc-600 rounded-full hover:bg-zinc-300 transition-colors"><div className="w-2.5 h-2.5 bg-current rounded-sm"/></button>
                      ) : (
                             <button onClick={() => handleSendMessage(activeSession.id)} disabled={!activeSession.inputValue.trim()} className="w-8 h-8 flex items-center justify-center bg-zinc-900 text-white rounded-full hover:scale-105 active:scale-95 disabled:opacity-30 disabled:scale-100 transition-all shadow-md shadow-zinc-300"><Play size={14} fill="currentColor" className="ml-0.5"/></button>
                      )}
                    </div>
                  </div>
                     {mentionOpen && (
                        <div className="absolute bottom-full left-4 right-4 mb-2 bg-white rounded-xl shadow-2xl border border-zinc-100 max-h-48 overflow-y-auto py-1 z-50">
                          {getColumnsList(activeSession).filter(c => !mentionQuery || String(c).toLowerCase().includes(mentionQuery.toLowerCase())).map(c => (
                            <button key={c} onClick={() => insertMention(c)} className="w-full text-left px-4 py-2 text-sm text-zinc-700 hover:bg-indigo-50 hover:text-indigo-700 transition-colors">@{c}</button>
                          ))}
                </div>
                     )}
              </div>
                </>
              ) : (
                <button onClick={() => setIsChatOpen(true)} className="h-full w-full flex items-center justify-center text-zinc-400 hover:bg-zinc-50" title="展开聊天">
                  <ChevronLeft size={18}/>
                </button>
              )}
           </div>
        </div>
      </div>

      {toolDrawerOpen && (
        <div className="absolute inset-0 z-50 overflow-hidden">
          <div className="absolute inset-0 bg-zinc-900/10 backdrop-blur-sm transition-opacity" onClick={() => setToolDrawerOpen(false)}/>
          <div className="absolute right-0 top-0 h-full w-[380px] bg-white shadow-2xl border-l border-zinc-200/60 flex flex-col transform transition-transform duration-300 animate-in slide-in-from-right">
            <div className="p-5 border-b border-zinc-100 flex items-center justify-between bg-white/50 backdrop-blur-md z-10">
              <div className="font-bold text-lg text-zinc-800">工具箱</div>
              <button onClick={() => setToolDrawerOpen(false)} className="p-1 text-zinc-400 hover:text-zinc-900 rounded-lg hover:bg-zinc-100"><X size={20}/></button>
            </div>
            <div className="px-5 pt-4 pb-2">
              <div className="flex p-1 bg-zinc-100 rounded-xl">
              {[
                { key: 'column', label: '列属性' },
                  { key: 'history', label: '历史栈' },
                  { key: 'clean', label: '清洗' },
                  { key: 'analysis', label: '分析库' },
              ].map(t => (
                  <button key={t.key} onClick={() => setToolTab(t.key)} className={`flex-1 py-1.5 text-xs font-medium rounded-lg transition-all ${toolTab === t.key ? 'bg-white text-indigo-600 shadow-sm' : 'text-zinc-500 hover:text-zinc-900'}`}>{t.label}</button>
              ))}
            </div>
            </div>
            <div className="flex-1 overflow-y-auto p-5 space-y-6">
              {toolTab === 'column' && (
                <div className="space-y-4">
                  <div className="text-xs text-slate-500">提示：点击数据表的列头即可打开对应列属性。</div>
                  {!selectedColumn ? (
                    <div className="text-sm text-slate-600 bg-slate-50 border border-slate-200 rounded p-3">未选择列。请先点击表头。</div>
                  ) : (
              <div className="space-y-4">
                 <div>
                        <div className="text-xs font-bold text-slate-600 mb-1">当前列</div>
                        <div className="text-sm font-mono text-slate-900 bg-slate-50 border border-slate-200 rounded px-3 py-2">{selectedColumn}</div>
                      </div>

                      <div>
                        <div className="text-xs font-bold text-slate-600 mb-1">重命名（数据变更，会进入历史栈）</div>
                    <div className="flex gap-2">
                          <input value={renameDraft} onChange={(e) => setRenameDraft(e.target.value)} className="flex-1 border rounded px-2 py-1 text-sm" />
                          <button
                            onClick={() => {
                              const next = (renameDraft || '').trim();
                              if (!next || next === selectedColumn) return;
                              applyActionsDirect(activeSession.id, [{ type: 'rename_columns', params: { mapping: { [selectedColumn]: next } } }], `已重命名列：${selectedColumn} → ${next}`);
                              setSelectedColumn(next);
                            }}
                            className="px-3 py-1 rounded bg-slate-900 text-white text-xs hover:bg-slate-800"
                          >
                            应用
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <div className="text-xs font-bold text-slate-600 mb-1">变量标签</div>
                          <input value={colMetaDraft.label} onChange={(e) => setColMetaDraft(prev => ({ ...prev, label: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm" placeholder="例如：孕妇BMI" />
                        </div>
                        <div>
                          <div className="text-xs font-bold text-slate-600 mb-1">度量类型</div>
                          <select value={colMetaDraft.measure} onChange={(e) => setColMetaDraft(prev => ({ ...prev, measure: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm">
                            <option value="scale">scale（连续）</option>
                            <option value="ordinal">ordinal（顺序）</option>
                            <option value="nominal">nominal（名义）</option>
                          </select>
                        </div>
                      </div>

                      <div>
                        <div className="text-xs font-bold text-slate-600 mb-1">缺失码（元数据，不会自动改数据）</div>
                        <input value={colMetaDraft.missingCodesText} onChange={(e) => setColMetaDraft(prev => ({ ...prev, missingCodesText: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm" placeholder="例如：999, 拒绝回答" />
                        <div className="mt-2 flex gap-2">
                          <button
                            onClick={() => {
                              const vals = parseMissingCodesText(colMetaDraft.missingCodesText);
                              if (!vals.length) return alert('请先填写缺失码');
                              applyActionsDirect(activeSession.id, [{ type: 'replace_missing', params: { columns: [selectedColumn], values: vals } }], `已将缺失码设为缺失：${selectedColumn}`);
                            }}
                            className="px-3 py-1 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700"
                          >
                            将缺失码应用到数据
                          </button>
                        </div>
                      </div>

                      <div>
                        <div className="text-xs font-bold text-slate-600 mb-1">值标签（每行：值=标签）</div>
                        <textarea value={colMetaDraft.valueLabelsText} onChange={(e) => setColMetaDraft(prev => ({ ...prev, valueLabelsText: e.target.value }))} className="w-full border rounded px-2 py-2 text-sm font-mono h-28" placeholder={"1=男\n2=女"} />
                      </div>

                      <button onClick={() => handleSaveColumnMeta(activeSession.id)} className="w-full bg-slate-900 text-white py-2 rounded text-sm hover:bg-slate-800">保存列元数据</button>

                      <div className="pt-2 border-t border-slate-200">
                        <div className="text-xs font-bold text-slate-600 mb-2">快捷清洗（对当前列）</div>
                        <div className="grid grid-cols-2 gap-2">
                          <button onClick={() => applyActionsDirect(activeSession.id, [{ type: 'standardize', params: { columns: [selectedColumn], method: 'zscore' } }], `标准化：${selectedColumn}`)} className="px-3 py-2 rounded bg-indigo-50 text-indigo-700 text-xs hover:bg-indigo-100">标准化(z)</button>
                          <button onClick={() => applyActionsDirect(activeSession.id, [{ type: 'trim_whitespace', params: { columns: [selectedColumn] } }], `去除空格：${selectedColumn}`)} className="px-3 py-2 rounded bg-slate-50 text-slate-700 text-xs hover:bg-slate-100">去空格</button>
                          <button onClick={() => applyActionsDirect(activeSession.id, [{ type: 'dropna_rows', params: { columns: [selectedColumn], how: cleanDraft.dropnaHow } }], `删除缺失行（列：${selectedColumn}）`)} className="px-3 py-2 rounded bg-slate-50 text-slate-700 text-xs hover:bg-slate-100">删缺失行</button>
                          <button onClick={() => applyActionsDirect(activeSession.id, [{ type: 'cast_type', params: { column: selectedColumn, to: cleanDraft.castType } }], `类型转换：${selectedColumn} -> ${cleanDraft.castType}`)} className="px-3 py-2 rounded bg-slate-50 text-slate-700 text-xs hover:bg-slate-100">类型转换</button>
                        </div>
                        <div className="grid grid-cols-2 gap-3 mt-3">
                          <div>
                            <div className="text-[11px] font-bold text-slate-600 mb-1">删缺失行 how</div>
                            <select value={cleanDraft.dropnaHow} onChange={(e) => setCleanDraft(prev => ({ ...prev, dropnaHow: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                              <option value="any">any</option>
                              <option value="all">all</option>
                            </select>
                          </div>
                          <div>
                            <div className="text-[11px] font-bold text-slate-600 mb-1">转换类型 to</div>
                            <select value={cleanDraft.castType} onChange={(e) => setCleanDraft(prev => ({ ...prev, castType: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                              <option value="float">float</option>
                              <option value="int">int</option>
                              <option value="string">string</option>
                              <option value="category">category</option>
                              <option value="datetime">datetime</option>
                            </select>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {toolTab === 'history' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-bold text-slate-800">历史栈</div>
                    <div className="text-xs text-slate-500">{activeSession.history?.cursor || 0}/{activeSession.history?.total || 0}</div>
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => handleJumpToCursor(activeSession.id, 0)} className="px-3 py-1 rounded bg-slate-100 text-slate-700 text-xs hover:bg-slate-200">回到最初</button>
                    <button onClick={() => handleJumpToCursor(activeSession.id, activeSession.history?.total || 0)} className="px-3 py-1 rounded bg-slate-100 text-slate-700 text-xs hover:bg-slate-200">跳到最新</button>
                  </div>
                  {(!activeSession.historyStack || activeSession.historyStack.length === 0) ? (
                    <div className="text-sm text-slate-600 bg-slate-50 border border-slate-200 rounded p-3">暂无历史操作。通过“清洗建议/菜单化清洗/列重命名”等会生成历史。</div>
                  ) : (
                    <div className="space-y-2">
                      {activeSession.historyStack.map(item => (
                        <button
                          key={item.index}
                          onClick={() => handleJumpToCursor(activeSession.id, item.index)}
                          className={`w-full text-left px-3 py-2 rounded border text-xs ${item.applied ? 'bg-white border-slate-200' : 'bg-slate-50 border-slate-200 text-slate-500'} hover:bg-indigo-50`}
                        >
                          <div className="flex items-center justify-between">
                            <span className={`font-mono ${item.applied ? 'text-slate-800' : 'text-slate-500'}`}>#{item.index}</span>
                            <span className={`${item.applied ? 'text-green-600' : 'text-slate-400'}`}>{item.applied ? '已应用' : '未应用'}</span>
                          </div>
                          <div className="mt-1 whitespace-pre-wrap">{item.brief}</div>
                         </button>
                       ))}
                    </div>
                  )}
                 </div>
              )}

              {toolTab === 'clean' && (
                <div className="space-y-4">
                  <div className="text-sm font-bold text-slate-800">菜单化清洗</div>
                  <div className="text-xs text-slate-500">这些操作直接走 Action Pipeline，可撤销/重做/跳转。</div>

                  <div className="bg-slate-50 border border-slate-200 rounded p-3 space-y-3">
                    <div className="text-xs font-bold text-slate-700">全表操作</div>
                    <button onClick={() => applyActionsDirect(activeSession.id, [{ type: 'dropna_rows', params: { columns: [], how: cleanDraft.dropnaHow } }], `删除缺失行（how=${cleanDraft.dropnaHow}）`)} className="w-full px-3 py-2 rounded bg-white border border-slate-200 text-xs hover:bg-slate-100">删除缺失行</button>
                    <div>
                      <div className="text-[11px] font-bold text-slate-600 mb-1">缺失码（用于 replace_missing）</div>
                      <input value={cleanDraft.missingCodes} onChange={(e) => setCleanDraft(prev => ({ ...prev, missingCodes: e.target.value }))} className="w-full border rounded px-2 py-1 text-sm" placeholder="例如：999, 拒绝回答" />
                      <button onClick={() => applyActionsDirect(activeSession.id, [{ type: 'replace_missing', params: { columns: [], values: parseMissingCodesText(cleanDraft.missingCodes) } }], `将缺失码设为缺失（全表）`)} className="mt-2 w-full px-3 py-2 rounded bg-white border border-slate-200 text-xs hover:bg-slate-100">应用缺失码到全表</button>
                    </div>
                    <button onClick={() => applyActionsDirect(activeSession.id, [{ type: 'deduplicate', params: { subset: [], keep: cleanDraft.dedupKeep } }], `去重（keep=${cleanDraft.dedupKeep}）`)} className="w-full px-3 py-2 rounded bg-white border border-slate-200 text-xs hover:bg-slate-100">去重</button>
                    <button onClick={() => applyActionsDirect(activeSession.id, [{ type: 'trim_whitespace', params: { columns: [] } }], `去除空格（字符串列）`)} className="w-full px-3 py-2 rounded bg-white border border-slate-200 text-xs hover:bg-slate-100">去除空格（字符串列）</button>
                    <button onClick={() => { if (!selectedColumn) return alert('请先点击表头选择列'); const lower = Number(cleanDraft.winsorLower); const upper = Number(cleanDraft.winsorUpper); applyActionsDirect(activeSession.id, [{ type: 'winsorize', params: { columns: [selectedColumn], lower: isFinite(lower) ? lower : 0.01, upper: isFinite(upper) ? upper : 0.99 } }], `缩尾处理：${selectedColumn}`); }} className="w-full px-3 py-2 rounded bg-white border border-slate-200 text-xs hover:bg-slate-100">缩尾/截尾（按所选列）</button>
                    <button onClick={() => { if (!selectedColumn) return alert('请先点击表头选择列'); applyActionsDirect(activeSession.id, [{ type: 'one_hot_encode', params: { columns: [selectedColumn], drop_first: !!cleanDraft.oneHotDropFirst, prefix_sep: cleanDraft.oneHotPrefixSep || '=' } }], `虚拟变量转换(one-hot)：${selectedColumn}`); }} className="w-full px-3 py-2 rounded bg-white border border-slate-200 text-xs hover:bg-slate-100">虚拟变量转换(one-hot)（按所选列）</button>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-[11px] font-bold text-slate-600 mb-1">dropna how</div>
                        <select value={cleanDraft.dropnaHow} onChange={(e) => setCleanDraft(prev => ({ ...prev, dropnaHow: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                          <option value="any">any</option>
                          <option value="all">all</option>
                        </select>
                      </div>
                      <div>
                        <div className="text-[11px] font-bold text-slate-600 mb-1">去重 keep</div>
                        <select value={cleanDraft.dedupKeep} onChange={(e) => setCleanDraft(prev => ({ ...prev, dedupKeep: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                          <option value="first">first</option>
                          <option value="last">last</option>
                        </select>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-[11px] font-bold text-slate-600 mb-1">缩尾 lower</div>
                        <input value={cleanDraft.winsorLower} onChange={(e) => setCleanDraft(prev => ({ ...prev, winsorLower: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs" placeholder="0.01" />
                      </div>
                      <div>
                        <div className="text-[11px] font-bold text-slate-600 mb-1">缩尾 upper</div>
                        <input value={cleanDraft.winsorUpper} onChange={(e) => setCleanDraft(prev => ({ ...prev, winsorUpper: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs" placeholder="0.99" />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-[11px] font-bold text-slate-600 mb-1">one-hot drop_first</div>
                        <select value={cleanDraft.oneHotDropFirst ? '1' : '0'} onChange={(e) => setCleanDraft(prev => ({ ...prev, oneHotDropFirst: e.target.value === '1' }))} className="w-full border rounded px-2 py-1 text-xs">
                          <option value="0">false</option>
                          <option value="1">true</option>
                        </select>
                      </div>
                      <div>
                        <div className="text-[11px] font-bold text-slate-600 mb-1">one-hot 分隔符</div>
                        <input value={cleanDraft.oneHotPrefixSep} onChange={(e) => setCleanDraft(prev => ({ ...prev, oneHotPrefixSep: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs" placeholder="=" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-white border border-slate-200 rounded p-3">
                    <div className="text-xs font-bold text-slate-700 mb-2">提示</div>
                    <div className="text-xs text-slate-600">更精细的“按列标准化/类型转换/缺失码”建议：点击某列表头，在“列属性”页操作。</div>
                  </div>
                </div>
              )}

              {toolTab === 'analysis' && (
                 <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-bold text-slate-800">算法中心</div>
                    <div className="text-[10px] text-slate-500">表格 + 解释 + 图</div>
                      </div>

                  <input
                    value={analysisSearch}
                    onChange={(e) => setAnalysisSearch(e.target.value)}
                    placeholder="搜索算法（例如：卡方 / 回归 / PCA）"
                    className="w-full border border-slate-200 rounded px-2 py-1 text-sm"
                  />

                  <div className="bg-slate-50 border border-slate-200 rounded p-2 max-h-56 overflow-y-auto">
                    {(() => {
                      const q = (analysisSearch || '').trim().toLowerCase();
                      const items = (ANALYSIS_ALGOS || []).filter(a => {
                        const cat = Array.isArray(a.categoryPath) ? a.categoryPath : [];
                        const hay = `${a.name} ${cat.join('/')} ${a.analysis}`.toLowerCase();
                        return !q || hay.includes(q);
                      });
                      const tree = buildAnalysisTree(items);
                      const sortZh = (arr) => arr.sort((a, b) => String(a).localeCompare(String(b), 'zh-Hans-CN'));
                      const renderNode = (node, title, depth, keyPrefix) => {
                        if (!node) return null;
                        const cnt = countAnalysisTreeItems(node);
                        const open = depth === 0 && !q;
                        return (
                          <details key={keyPrefix} open={open} className="mb-1">
                            <summary className="cursor-pointer select-none text-xs font-bold text-slate-700 flex items-center justify-between px-2 py-1 rounded hover:bg-white">
                              <span className="truncate">{title}</span>
                              <span className="text-[10px] text-slate-400">{cnt}</span>
                            </summary>
                            <div className="pl-2 pt-1 space-y-1">
                              {sortZh(Object.keys(node.children || {})).map(k => renderNode(node.children[k], k, depth + 1, `${keyPrefix}/${k}`))}
                              {(node.items || []).map(algo => (
                                <button
                                  key={algo.id}
                                  onClick={() => selectAnalysisAlgo(algo.id)}
                                  disabled={algo.status !== 'ready'}
                                  className={`w-full text-left px-2 py-1 rounded text-xs border ${analysisSelectedId === algo.id ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'} disabled:opacity-50 disabled:cursor-not-allowed`}
                                  title={algo.status === 'ready' ? '点击配置参数并运行' : '敬请期待'}
                                >
                                  <div className="flex items-center justify-between">
                                    <span className="truncate">{algo.name}</span>
                                    {algo.status !== 'ready' && <span className="text-[10px]">敬请期待</span>}
                                  </div>
                                </button>
                    ))}
                 </div>
                          </details>
                        );
                      };
                      const keys = sortZh(Object.keys(tree.children || {}));
                      if (!keys.length) return <div className="text-xs text-slate-500 p-2">未找到匹配算法。</div>;
                      return keys.map(k => renderNode(tree.children[k], k, 0, k));
                    })()}
                  </div>

                  {(() => {
                    const algo = getAnalysisAlgo(analysisSelectedId);
                    if (!algo) return null;
                    const cols = getColumnsList(activeSession);
                    const form = analysisForms?.[algo.id] || makeDefaultAnalysisForm(algo);
                    const renderField = (f) => {
                      const v = form?.[f.key];
                      if (f.type === 'column') {
                        return (
                          <select
                            value={v || ''}
                            onChange={(e) => updateAnalysisFormField(algo.id, f.key, e.target.value)}
                            className="w-full border rounded px-2 py-1 text-xs"
                          >
                            <option value="">请选择</option>
                            {cols.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        );
                      }
                      if (f.type === 'columns') {
                        const selectedCols = Array.isArray(v) ? v : [];
                        return (
                          <div className="w-full border rounded px-2 py-2 text-xs max-h-32 overflow-y-auto bg-white">
                            {cols.length === 0 ? (
                              <div className="text-slate-400 py-1">暂无可用列</div>
                            ) : (
                              cols.map(c => (
                                <label key={c} className="flex items-center py-0.5 hover:bg-slate-50 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={selectedCols.includes(c)}
                                    onChange={(e) => {
                                      const newSelection = e.target.checked
                                        ? [...selectedCols, c]
                                        : selectedCols.filter(col => col !== c);
                                      updateAnalysisFormField(algo.id, f.key, newSelection);
                                    }}
                                    className="mr-2"
                                  />
                                  <span className="text-xs">{c}</span>
                                </label>
                              ))
                            )}
                            {f.required === false && (
                              <div className="text-[10px] text-slate-400 mt-1 pt-1 border-t border-slate-200">
                                留空=使用全部数值列
                              </div>
                            )}
                          </div>
                        );
                      }
                      if (f.type === 'number') {
                        return (
                          <input
                            value={v ?? ''}
                            onChange={(e) => updateAnalysisFormField(algo.id, f.key, e.target.value)}
                            className="w-full border rounded px-2 py-1 text-xs"
                            placeholder={String(f.default ?? '')}
                          />
                        );
                      }
                      if (f.type === 'select') {
                        const val = (v ?? (f.default ?? (f.options?.[0] ?? '')));
                        return (
                          <select
                            value={val}
                            onChange={(e) => updateAnalysisFormField(algo.id, f.key, e.target.value)}
                            className="w-full border rounded px-2 py-1 text-xs"
                          >
                            {(f.options || []).map(opt => <option key={opt} value={opt}>{opt}</option>)}
                          </select>
                        );
                      }
                      return (
                        <input
                          value={v ?? ''}
                          onChange={(e) => updateAnalysisFormField(algo.id, f.key, e.target.value)}
                          className="w-full border rounded px-2 py-1 text-xs"
                          placeholder={f.placeholder || ''}
                        />
                      );
                    };
                    return (
                      <div className="bg-white border border-slate-200 rounded p-3 space-y-3">
                        <div className="flex items-center justify-between gap-2">
                          <div className="text-xs font-bold text-slate-800 truncate">{algo.name}</div>
                          <button
                            onClick={runSelectedAnalysisAlgo}
                            disabled={algo.status !== 'ready'}
                            className="px-3 py-1 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700 disabled:opacity-50"
                          >
                            运行
                          </button>
                        </div>
                        {algo.status !== 'ready' ? (
                          <div className="text-xs text-slate-500">该算法暂未开放。</div>
                        ) : (
                          <>
                            {(!algo.fields || algo.fields.length === 0) && (
                              <div className="text-xs text-slate-500">无需参数，直接点击“运行”。</div>
                            )}
                            <div className="space-y-3">
                              {(algo.fields || []).map(f => (
                                <div key={f.key}>
                                  <div className="text-[11px] font-bold text-slate-600 mb-1">{f.label}</div>
                                  {renderField(f)}
                                </div>
                              ))}
                            </div>
                            <div className="text-[10px] text-slate-500 leading-relaxed">
                              提示：点击数据表列头可快速选中列；部分算法支持自动使用“所选列”。
                            </div>
                          </>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}
              {/* 旧版“分析”面板已废弃：保留代码仅用于回溯（勿删，后续会彻底移除）
              {toolTab === 'analysis' && (
                <div className="space-y-4">
                  <div className="text-sm font-bold text-slate-800">统计/建模分析</div>
                  <div className="text-xs text-slate-500">输出：表格（可复制）+ DeepSeek 解释 + 图表（产物存放到 out/）。</div>

                  <details className="bg-white border border-slate-200 rounded p-3" open>
                    <summary className="cursor-pointer text-xs font-bold text-slate-700">1) 描述性分析</summary>
                    <div className="mt-3 space-y-3">
                      <button onClick={() => runAnalysis(activeSession.id, 'overview', {}, '【分析】数据概览')} className="w-full px-3 py-2 rounded bg-slate-900 text-white text-xs hover:bg-slate-800">数据概览</button>

                      <div className="grid grid-cols-2 gap-2">
                        <button onClick={() => runAnalysis(activeSession.id, 'descriptive', { columns: analysisDraft.descCols || [] }, '【分析】描述统计')} className="px-3 py-2 rounded bg-slate-100 text-slate-700 text-xs hover:bg-slate-200">描述统计（选中列）</button>
                        <button onClick={() => runAnalysis(activeSession.id, 'descriptive', { columns: [] }, '【分析】描述统计（全数值列）')} className="px-3 py-2 rounded bg-slate-100 text-slate-700 text-xs hover:bg-slate-200">描述统计（全数值列）</button>
                      </div>
                      <div>
                        <div className="text-[11px] font-bold text-slate-600 mb-1">描述统计：选择列（可多选）</div>
                        <div className="w-full border rounded px-2 py-2 text-xs max-h-32 overflow-y-auto bg-white">
                          {getColumnsList(activeSession).map(c => (
                            <label key={c} className="flex items-center py-0.5 hover:bg-slate-50 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={(analysisDraft.descCols || []).includes(c)}
                                onChange={(e) => {
                                  const current = analysisDraft.descCols || [];
                                  const newCols = e.target.checked
                                    ? [...current, c]
                                    : current.filter(col => col !== c);
                                  setAnalysisDraft(prev => ({ ...prev, descCols: newCols }));
                                }}
                                className="mr-2"
                              />
                              <span className="text-xs">{c}</span>
                            </label>
                          ))}
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-2">
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">频数分析：列</div>
                          <select value={(analysisDraft.freqCol || selectedColumn || '')} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, freqCol: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div className="flex items-end">
                          <button onClick={() => { const col = analysisDraft.freqCol || selectedColumn; if (!col) return alert('请先选择列'); runAnalysis(activeSession.id, 'frequency', { column: col, top_n: 30 }, `【分析】频数分析：${col}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行</button>
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">列联：行变量</div>
                          <select value={analysisDraft.crossRow} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, crossRow: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">列联：列变量</div>
                          <select value={analysisDraft.crossCol} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, crossCol: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div className="flex items-end">
                          <button onClick={() => { if (!analysisDraft.crossRow || !analysisDraft.crossCol) return alert('请选择行/列变量'); runAnalysis(activeSession.id, 'crosstab', { row: analysisDraft.crossRow, col: analysisDraft.crossCol }, `【分析】列联分析：${analysisDraft.crossRow} × ${analysisDraft.crossCol}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行</button>
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">分类汇总：分组列</div>
                          <select value={analysisDraft.groupBy} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, groupBy: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">分类汇总：指标列</div>
                          <select value={analysisDraft.groupMetric} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, groupMetric: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">聚合</div>
                          <select value={analysisDraft.groupAgg} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, groupAgg: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="mean">mean</option>
                            <option value="median">median</option>
                            <option value="sum">sum</option>
                            <option value="count">count</option>
                          </select>
                        </div>
                      </div>
                      <button onClick={() => { if (!analysisDraft.groupBy || !analysisDraft.groupMetric) return alert('请选择分组列与指标列'); runAnalysis(activeSession.id, 'group_summary', { group_by: analysisDraft.groupBy, metric: analysisDraft.groupMetric, agg: analysisDraft.groupAgg }, `【分析】分类汇总：${analysisDraft.groupBy} → ${analysisDraft.groupAgg}(${analysisDraft.groupMetric})`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行分类汇总</button>

                      <div className="grid grid-cols-3 gap-2">
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">正态性检验：列</div>
                          <select value={(analysisDraft.normalCol || selectedColumn || '')} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, normalCol: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">方法</div>
                          <select value={analysisDraft.normalMethod} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, normalMethod: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="auto">auto</option>
                            <option value="shapiro">shapiro</option>
                            <option value="normaltest">normaltest</option>
                            <option value="jarque_bera">jarque_bera</option>
                          </select>
                        </div>
                      </div>
                      <button onClick={() => { const col = analysisDraft.normalCol || selectedColumn; if (!col) return alert('请选择列'); runAnalysis(activeSession.id, 'normality', { column: col, method: analysisDraft.normalMethod }, `【分析】正态性检验：${col}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行正态性检验</button>
                    </div>
                  </details>

                  <details className="bg-white border border-slate-200 rounded p-3">
                    <summary className="cursor-pointer text-xs font-bold text-slate-700">2) 差异性分析</summary>
                    <div className="mt-3 space-y-3">
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">T检验类型</div>
                          <select value={analysisDraft.tType} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, tType: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="one_sample">单样本T</option>
                            <option value="independent">独立样本T</option>
                            <option value="paired">配对样本T</option>
                          </select>
                        </div>
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">目标列 y</div>
                          <select value={analysisDraft.tY} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, tY: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                      </div>

                      {analysisDraft.tType === 'one_sample' && (
                        <div className="grid grid-cols-3 gap-2">
                          <div className="col-span-2">
                            <div className="text-[11px] font-bold text-slate-600 mb-1">mu（检验均值）</div>
                            <input value={analysisDraft.tMu} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, tMu: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs" placeholder="0" />
                          </div>
                          <div className="flex items-end">
                            <button onClick={() => { if (!analysisDraft.tY) return alert('请选择 y'); runAnalysis(activeSession.id, 'ttest', { ttype: 'one_sample', y: analysisDraft.tY, mu: Number(analysisDraft.tMu || 0) }, `【分析】单样本T：${analysisDraft.tY}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行</button>
                          </div>
                        </div>
                      )}

                      {analysisDraft.tType === 'paired' && (
                        <div className="grid grid-cols-3 gap-2">
                          <div className="col-span-2">
                            <div className="text-[11px] font-bold text-slate-600 mb-1">配对列 y2</div>
                            <select value={analysisDraft.tY2} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, tY2: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                              <option value="">请选择</option>
                              {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                          </div>
                          <div className="flex items-end">
                            <button onClick={() => { if (!analysisDraft.tY || !analysisDraft.tY2) return alert('请选择 y 与 y2'); runAnalysis(activeSession.id, 'ttest', { ttype: 'paired', y: analysisDraft.tY, y2: analysisDraft.tY2 }, `【分析】配对T：${analysisDraft.tY} vs ${analysisDraft.tY2}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行</button>
                          </div>
                        </div>
                      )}

                      {analysisDraft.tType === 'independent' && (
                        <div className="space-y-2">
                          <div className="grid grid-cols-3 gap-2">
                            <div className="col-span-2">
                              <div className="text-[11px] font-bold text-slate-600 mb-1">分组列 group_col</div>
                              <select value={analysisDraft.tGroup} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, tGroup: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                                <option value="">请选择</option>
                                {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                              </select>
                            </div>
                            <div className="flex items-end">
                              <button onClick={() => { if (!analysisDraft.tY || !analysisDraft.tGroup) return alert('请选择 y 与 group_col'); runAnalysis(activeSession.id, 'ttest', { ttype: 'independent', y: analysisDraft.tY, group_col: analysisDraft.tGroup, group_a: analysisDraft.tA || null, group_b: analysisDraft.tB || null }, `【分析】独立样本T：${analysisDraft.tY} by ${analysisDraft.tGroup}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行</button>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <div className="text-[11px] font-bold text-slate-600 mb-1">组A（可选）</div>
                              <input value={analysisDraft.tA} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, tA: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs" placeholder="留空=自动选频数最高的两组" />
                            </div>
                            <div>
                              <div className="text-[11px] font-bold text-slate-600 mb-1">组B（可选）</div>
                              <input value={analysisDraft.tB} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, tB: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs" />
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="grid grid-cols-3 gap-2">
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">单因素ANOVA：y</div>
                          <select value={analysisDraft.anovaY} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, anovaY: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">分组列</div>
                          <select value={analysisDraft.anovaGroup} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, anovaGroup: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                      </div>
                      <button onClick={() => { if (!analysisDraft.anovaY || !analysisDraft.anovaGroup) return alert('请选择 y 与分组列'); runAnalysis(activeSession.id, 'anova', { y: analysisDraft.anovaY, group_col: analysisDraft.anovaGroup }, `【分析】单因素ANOVA：${analysisDraft.anovaY} by ${analysisDraft.anovaGroup}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行ANOVA</button>

                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">卡方：row</div>
                          <select value={analysisDraft.chiRow} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, chiRow: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">卡方：col</div>
                          <select value={analysisDraft.chiCol} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, chiCol: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div className="flex items-end">
                          <button onClick={() => { if (!analysisDraft.chiRow || !analysisDraft.chiCol) return alert('请选择 row/col'); runAnalysis(activeSession.id, 'chi_square', { row: analysisDraft.chiRow, col: analysisDraft.chiCol }, `【分析】卡方检验：${analysisDraft.chiRow} × ${analysisDraft.chiCol}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行</button>
                        </div>
                      </div>

                      <div className="bg-slate-50 border border-slate-200 rounded p-3 space-y-2">
                        <div className="text-xs font-bold text-slate-700">非参数检验</div>
                        <div className="grid grid-cols-3 gap-2">
                          <div>
                            <div className="text-[11px] font-bold text-slate-600 mb-1">类型</div>
                            <select value={analysisDraft.nonparamTest} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, nonparamTest: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                              <option value="mann_whitney">Mann-Whitney</option>
                              <option value="kruskal">Kruskal-Wallis</option>
                              <option value="friedman">Friedman</option>
                            </select>
                          </div>
                          <div className="col-span-2">
                            {analysisDraft.nonparamTest !== 'friedman' ? (
                              <>
                                <div className="text-[11px] font-bold text-slate-600 mb-1">y</div>
                                <select value={analysisDraft.nonparamY} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, nonparamY: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                                  <option value="">请选择</option>
                                  {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                              </>
                            ) : (
                              <>
                                <div className="text-[11px] font-bold text-slate-600 mb-1">列（可多选，≥3）</div>
                                <div className="w-full border rounded px-2 py-2 text-xs max-h-32 overflow-y-auto bg-white">
                                  {getColumnsList(activeSession).map(c => (
                                    <label key={c} className="flex items-center py-0.5 hover:bg-slate-50 cursor-pointer">
                                      <input
                                        type="checkbox"
                                        checked={(analysisDraft.nonparamCols || []).includes(c)}
                                        onChange={(e) => {
                                          const current = analysisDraft.nonparamCols || [];
                                          const newCols = e.target.checked
                                            ? [...current, c]
                                            : current.filter(col => col !== c);
                                          setAnalysisDraft(prev => ({ ...prev, nonparamCols: newCols }));
                                        }}
                                        className="mr-2"
                                      />
                                      <span className="text-xs">{c}</span>
                                    </label>
                                  ))}
                                </div>
                              </>
                            )}
                          </div>
                        </div>
                        {analysisDraft.nonparamTest !== 'friedman' && (
                          <div className="grid grid-cols-3 gap-2">
                            <div className="col-span-2">
                              <div className="text-[11px] font-bold text-slate-600 mb-1">分组列 group_col</div>
                              <select value={analysisDraft.nonparamGroup} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, nonparamGroup: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                                <option value="">请选择</option>
                                {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                              </select>
                            </div>
                            <div className="flex items-end">
                              <button onClick={() => { if (!analysisDraft.nonparamY || !analysisDraft.nonparamGroup) return alert('请选择 y 与 group_col'); runAnalysis(activeSession.id, 'nonparam', { test: analysisDraft.nonparamTest, y: analysisDraft.nonparamY, group_col: analysisDraft.nonparamGroup }, `【分析】非参(${analysisDraft.nonparamTest})：${analysisDraft.nonparamY} by ${analysisDraft.nonparamGroup}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行</button>
                            </div>
                          </div>
                        )}
                        {analysisDraft.nonparamTest === 'friedman' && (
                          <button onClick={() => { if (!analysisDraft.nonparamCols || analysisDraft.nonparamCols.length < 3) return alert('请选择至少3列'); runAnalysis(activeSession.id, 'nonparam', { test: 'friedman', columns: analysisDraft.nonparamCols }, `【分析】Friedman：${analysisDraft.nonparamCols.join(', ')}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行Friedman</button>
                        )}
                      </div>
                    </div>
                  </details>

                  <details className="bg-white border border-slate-200 rounded p-3">
                    <summary className="cursor-pointer text-xs font-bold text-slate-700">3) 相关/回归/建模</summary>
                    <div className="mt-3 space-y-3">
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">相关方法</div>
                          <select value={analysisDraft.corrMethod} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, corrMethod: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="pearson">Pearson</option>
                            <option value="spearman">Spearman</option>
                            <option value="kendall">Kendall</option>
                          </select>
                        </div>
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">列（可多选，留空=全数值列）</div>
                          <div className="w-full border rounded px-2 py-2 text-xs max-h-32 overflow-y-auto bg-white">
                            {getColumnsList(activeSession).map(c => (
                              <label key={c} className="flex items-center py-0.5 hover:bg-slate-50 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={(analysisDraft.corrCols || []).includes(c)}
                                  onChange={(e) => {
                                    const current = analysisDraft.corrCols || [];
                                    const newCols = e.target.checked
                                      ? [...current, c]
                                      : current.filter(col => col !== c);
                                    setAnalysisDraft(prev => ({ ...prev, corrCols: newCols }));
                                  }}
                                  className="mr-2"
                                />
                                <span className="text-xs">{c}</span>
                              </label>
                            ))}
                            <div className="text-[10px] text-slate-400 mt-1 pt-1 border-t border-slate-200">
                              留空=使用全部数值列
                            </div>
                          </div>
                        </div>
                      </div>
                      <button onClick={() => runAnalysis(activeSession.id, 'correlation', { method: analysisDraft.corrMethod, columns: analysisDraft.corrCols || [] }, `【分析】相关性分析(${analysisDraft.corrMethod})`)} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行相关性分析</button>

                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">线性回归 y</div>
                          <select value={analysisDraft.linregY} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, linregY: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">线性回归 X（可多选）</div>
                          <div className="w-full border rounded px-2 py-2 text-xs max-h-32 overflow-y-auto bg-white">
                            {getColumnsList(activeSession).map(c => (
                              <label key={c} className="flex items-center py-0.5 hover:bg-slate-50 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={(analysisDraft.linregX || []).includes(c)}
                                  onChange={(e) => {
                                    const current = analysisDraft.linregX || [];
                                    const newCols = e.target.checked
                                      ? [...current, c]
                                      : current.filter(col => col !== c);
                                    setAnalysisDraft(prev => ({ ...prev, linregX: newCols }));
                                  }}
                                  className="mr-2"
                                />
                                <span className="text-xs">{c}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      </div>
                      <button onClick={() => { if (!analysisDraft.linregY || !analysisDraft.linregX.length) return alert('请选择 y 与 X'); runAnalysis(activeSession.id, 'linear_regression', { y: analysisDraft.linregY, x: analysisDraft.linregX }, `【分析】线性回归：${analysisDraft.linregY} ~ ${analysisDraft.linregX.join('+')}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行线性回归</button>

                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">Logistic y（二元）</div>
                          <select value={analysisDraft.logitY} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, logitY: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs">
                            <option value="">请选择</option>
                            {getColumnsList(activeSession).map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        </div>
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">Logistic X（可多选）</div>
                          <div className="w-full border rounded px-2 py-2 text-xs max-h-32 overflow-y-auto bg-white">
                            {getColumnsList(activeSession).map(c => (
                              <label key={c} className="flex items-center py-0.5 hover:bg-slate-50 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={(analysisDraft.logitX || []).includes(c)}
                                  onChange={(e) => {
                                    const current = analysisDraft.logitX || [];
                                    const newCols = e.target.checked
                                      ? [...current, c]
                                      : current.filter(col => col !== c);
                                    setAnalysisDraft(prev => ({ ...prev, logitX: newCols }));
                                  }}
                                  className="mr-2"
                                />
                                <span className="text-xs">{c}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      </div>
                      <button onClick={() => { if (!analysisDraft.logitY || !analysisDraft.logitX.length) return alert('请选择 y 与 X'); runAnalysis(activeSession.id, 'logistic_regression', { y: analysisDraft.logitY, x: analysisDraft.logitX }, `【分析】Logistic回归：${analysisDraft.logitY}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行Logistic回归</button>

                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">PCA 组件数</div>
                          <input value={analysisDraft.pcaN} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, pcaN: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs" placeholder="2" />
                        </div>
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">PCA 列（可多选）</div>
                          <div className="w-full border rounded px-2 py-2 text-xs max-h-32 overflow-y-auto bg-white">
                            {getColumnsList(activeSession).map(c => (
                              <label key={c} className="flex items-center py-0.5 hover:bg-slate-50 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={(analysisDraft.pcaCols || []).includes(c)}
                                  onChange={(e) => {
                                    const current = analysisDraft.pcaCols || [];
                                    const newCols = e.target.checked
                                      ? [...current, c]
                                      : current.filter(col => col !== c);
                                    setAnalysisDraft(prev => ({ ...prev, pcaCols: newCols }));
                                  }}
                                  className="mr-2"
                                />
                                <span className="text-xs">{c}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      </div>
                      <button onClick={() => { if (!analysisDraft.pcaCols.length) return alert('请选择列'); runAnalysis(activeSession.id, 'pca', { columns: analysisDraft.pcaCols, n_components: Number(analysisDraft.pcaN || 2) }, `【分析】PCA：${analysisDraft.pcaCols.join(', ')}`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行PCA</button>

                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <div className="text-[11px] font-bold text-slate-600 mb-1">KMeans k</div>
                          <input value={analysisDraft.kmeansK} onChange={(e) => setAnalysisDraft(prev => ({ ...prev, kmeansK: e.target.value }))} className="w-full border rounded px-2 py-1 text-xs" placeholder="3" />
                        </div>
                        <div className="col-span-2">
                          <div className="text-[11px] font-bold text-slate-600 mb-1">KMeans 列（可多选）</div>
                          <div className="w-full border rounded px-2 py-2 text-xs max-h-32 overflow-y-auto bg-white">
                            {getColumnsList(activeSession).map(c => (
                              <label key={c} className="flex items-center py-0.5 hover:bg-slate-50 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={(analysisDraft.kmeansCols || []).includes(c)}
                                  onChange={(e) => {
                                    const current = analysisDraft.kmeansCols || [];
                                    const newCols = e.target.checked
                                      ? [...current, c]
                                      : current.filter(col => col !== c);
                                    setAnalysisDraft(prev => ({ ...prev, kmeansCols: newCols }));
                                  }}
                                  className="mr-2"
                                />
                                <span className="text-xs">{c}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      </div>
                      <button onClick={() => { if (!analysisDraft.kmeansCols.length) return alert('请选择列'); runAnalysis(activeSession.id, 'kmeans', { columns: analysisDraft.kmeansCols, k: Number(analysisDraft.kmeansK || 3) }, `【分析】KMeans(k=${analysisDraft.kmeansK})`); }} className="w-full px-3 py-2 rounded bg-indigo-600 text-white text-xs hover:bg-indigo-700">运行KMeans</button>
                    </div>
                  </details>
                </div>
              )}
              */}
            </div>
          </div>
        </div>
      )}

      {previewImage && (
        <div className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-xl flex items-center justify-center p-8 animate-in fade-in duration-200" onClick={()=>setPreviewImage(null)}>
          <img alt="" src={previewImage} className="max-h-full max-w-full rounded-lg shadow-2xl" onClick={e=>e.stopPropagation()}/>
          <button className="absolute top-8 right-8 text-white/50 hover:text-white"><X size={32}/></button>
        </div>
      )}
      
      {showSettings && (
        <div className="absolute inset-0 z-[60] flex items-center justify-center p-4">
           <div className="absolute inset-0 bg-black/20 backdrop-blur-sm transition-opacity" onClick={() => setShowSettings(false)}/>
           <div className="bg-white rounded-3xl shadow-2xl w-full max-w-md p-6 relative z-10 animate-in zoom-in-95 duration-200">
              <h3 className="text-xl font-bold text-zinc-900 mb-6 flex items-center gap-2"><Settings className="text-zinc-400"/> 系统设置</h3>
              <div className="space-y-4">
                 <div className="space-y-3">
                    <label className="block text-xs font-bold text-slate-500">API Keys（全局，跨任务通用）</label>
                    <div className="flex items-center gap-2">
                      <span className="w-24 text-xs font-bold text-slate-700">DeepSeek</span>
                      <input
                        type="password"
                        value={globalDeepSeekKey || ''}
                        onChange={(e) => setGlobalDeepSeekKey(e.target.value)}
                        className="flex-1 border rounded px-2 py-1 text-sm"
                        placeholder="sk-..."
                      />
                    </div>
                    <div className="text-[10px] text-slate-500 leading-relaxed">
                      <div>该 Key 会自动用于 DeepSeek-A / DeepSeek-B / DeepSeek-C，并在多个任务窗口通用；新建任务无需重复配置。</div>
                    </div>

                    <div className="flex items-center gap-2 mt-2">
                      <span className="w-24 text-xs font-bold text-slate-700">智谱(Zhipu)</span>
                      <input
                        type="password"
                        value={globalZhipuKey || ''}
                        onChange={(e) => setGlobalZhipuKey(e.target.value)}
                        className="flex-1 border rounded px-2 py-1 text-sm"
                        placeholder="glm-... key"
                      />
                    </div>

                    <div className="flex items-center gap-2 mt-2">
                      <span className="w-24 text-xs font-bold text-slate-700">千问(Qwen)</span>
                      <input
                        type="password"
                        value={globalQwenKey || ''}
                        onChange={(e) => setGlobalQwenKey(e.target.value)}
                        className="flex-1 border rounded px-2 py-1 text-sm"
                        placeholder="dashscope key"
                      />
                    </div>

                    <div className="text-[10px] text-slate-500 leading-relaxed">
                      <div className="mt-1">提示：Ask/Agent 模式切换在输入栏完成；这里配置全局 API Key 与当前任务的模型偏好。</div>
                    </div>
                 </div>

                 <div className="pt-3 border-t border-slate-200 space-y-3">
                   <label className="block text-xs font-bold text-slate-500">模型偏好（按模式）</label>

                   {/* Ask */}
                   <div className="bg-slate-50 border border-slate-200 rounded p-3 space-y-2">
                     <div className="text-xs font-bold text-slate-700">Ask（单API）</div>
                     {(() => {
                       const ms = activeSession.modelPrefs?.ask || { provider: 'deepseekA', model: 'deepseek-chat' };
                       const p = ms.provider || 'deepseekA';
                       const opts = getProviderModelOptions(p);
                       const m = ms.model && opts.includes(ms.model) ? ms.model : (getDefaultModelFor(p, 'ask') || opts[0] || '');
                       return (
                         <div className="grid grid-cols-2 gap-2">
                           <select value={p} onChange={(e) => updateSession(activeSession.id, { modelPrefs: { ...(activeSession.modelPrefs||{}), ask: { provider: e.target.value, model: getDefaultModelFor(e.target.value, 'ask') } } })} className="w-full border rounded px-2 py-1 text-xs bg-white">
                             {allProviderIds.map(pid => <option key={pid} value={pid}>{PROVIDER_LABELS[pid] || pid}</option>)}
                           </select>
                           <select value={m} onChange={(e) => updateSession(activeSession.id, { modelPrefs: { ...(activeSession.modelPrefs||{}), ask: { provider: p, model: e.target.value } } })} className="w-full border rounded px-2 py-1 text-xs bg-white">
                             {opts.map(mm => <option key={mm} value={mm}>{mm}</option>)}
                           </select>
                         </div>
                       );
                     })()}
                   </div>

                   {/* Agent 单模型 */}
                   <div className="bg-slate-50 border border-slate-200 rounded p-3 space-y-2">
                     <div className="text-xs font-bold text-slate-700">Agent（单模型）</div>
                     {(() => {
                       const ms = activeSession.modelPrefs?.agent_single || { provider: 'deepseekA', model: 'deepseek-reasoner' };
                       const p = ms.provider || 'deepseekA';
                       const opts = getProviderModelOptions(p);
                       const m = ms.model && opts.includes(ms.model) ? ms.model : (getDefaultModelFor(p, 'agent_single') || opts[0] || '');
                       return (
                         <div className="grid grid-cols-2 gap-2">
                           <select value={p} onChange={(e) => updateSession(activeSession.id, { modelPrefs: { ...(activeSession.modelPrefs||{}), agent_single: { provider: e.target.value, model: getDefaultModelFor(e.target.value, 'agent_single') } } })} className="w-full border rounded px-2 py-1 text-xs bg-white">
                             {allProviderIds.map(pid => <option key={pid} value={pid}>{PROVIDER_LABELS[pid] || pid}</option>)}
                           </select>
                           <select value={m} onChange={(e) => updateSession(activeSession.id, { modelPrefs: { ...(activeSession.modelPrefs||{}), agent_single: { provider: p, model: e.target.value } } })} className="w-full border rounded px-2 py-1 text-xs bg-white">
                             {opts.map(mm => <option key={mm} value={mm}>{mm}</option>)}
                           </select>
                         </div>
                       );
                     })()}
                   </div>

                   {/* Agent 多专家 */}
                   <div className="bg-slate-50 border border-slate-200 rounded p-3 space-y-2">
                     <div className="text-xs font-bold text-slate-700">Agent（多专家）</div>
                     {(['planner','executor','verifier']).map(role => {
                       const roleLabel = role === 'planner' ? '规划' : role === 'executor' ? '执行' : '评审';
                       const r = activeSession.modelPrefs?.agent_multi?.[role] || { provider: role==='planner'?'deepseekA':role==='executor'?'deepseekB':'deepseekC', model: 'deepseek-reasoner' };
                       const p = r.provider;
                       const opts = getProviderModelOptions(p);
                       const m = r.model && opts.includes(r.model) ? r.model : (getDefaultModelFor(p, 'agent_single') || opts[0] || '');
                       return (
                         <div key={role} className="grid grid-cols-3 gap-2 items-center">
                           <div className="text-[11px] font-bold text-slate-600">{roleLabel}</div>
                           <select value={p} onChange={(e) => updateSession(activeSession.id, { modelPrefs: { ...(activeSession.modelPrefs||{}), agent_multi: { ...(activeSession.modelPrefs?.agent_multi||{}), [role]: { provider: e.target.value, model: getDefaultModelFor(e.target.value, 'agent_single') } } } })} className="w-full border rounded px-2 py-1 text-xs bg-white">
                             {allProviderIds.map(pid => <option key={pid} value={pid}>{PROVIDER_LABELS[pid] || pid}</option>)}
                           </select>
                           <select value={m} onChange={(e) => updateSession(activeSession.id, { modelPrefs: { ...(activeSession.modelPrefs||{}), agent_multi: { ...(activeSession.modelPrefs?.agent_multi||{}), [role]: { provider: p, model: e.target.value } } } })} className="w-full border rounded px-2 py-1 text-xs bg-white">
                             {opts.map(mm => <option key={mm} value={mm}>{mm}</option>)}
                           </select>
                         </div>
                       );
                     })}
                   </div>
                 </div>

                 <button onClick={()=>setShowSettings(false)} className="w-full py-3 bg-zinc-900 text-white rounded-xl font-semibold hover:bg-zinc-800 transition-transform active:scale-95 shadow-lg shadow-zinc-200">保存并关闭</button>
              </div>
           </div>
        </div>
      )}

      {showDbModal && (
        <div className="absolute inset-0 z-[60] flex items-center justify-center p-4">
           <div className="absolute inset-0 bg-black/20 backdrop-blur-sm" onClick={() => setShowDbModal(false)}/>
           <div className="bg-white rounded-3xl shadow-2xl w-full max-w-md p-8 relative z-10 animate-in zoom-in-95 duration-200">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-zinc-900 flex items-center gap-2"><Server className="text-indigo-500"/> 连接数据库</h3>
                <button onClick={() => setShowDbModal(false)} className="p-1 rounded-full hover:bg-zinc-100"><X size={20}/></button>
              </div>
              <div className="space-y-4">
                 <div className="flex p-1 bg-zinc-100 rounded-xl">
                    {['mysql','postgres','sqlite'].map(t=><button key={t} onClick={()=>setDbConfig({...dbConfig, type:t})} className={`flex-1 py-2 text-xs font-medium rounded-lg transition-all ${dbConfig.type===t?'bg-white text-indigo-600 shadow-sm':'text-zinc-500'}`}>{t.toUpperCase()}</button>)}
                 </div>
                 {dbConfig.type!=='sqlite' && <input value={dbConfig.host} onChange={e=>setDbConfig({...dbConfig, host:e.target.value})} placeholder="Host (e.g., localhost)" className="w-full px-4 py-2.5 bg-zinc-50 border border-zinc-200 rounded-xl text-sm focus:bg-white"/>}
                 <input value={dbConfig.database} onChange={e=>setDbConfig({...dbConfig, database:e.target.value})} placeholder={dbConfig.type==='sqlite'?"Path (e.g., /data/db.sqlite)":"Database Name"} className="w-full px-4 py-2.5 bg-zinc-50 border border-zinc-200 rounded-xl text-sm focus:bg-white"/>
                 <div className="relative">
                   <div className="absolute top-2.5 left-4 text-xs font-bold text-zinc-400">SQL</div>
                   <textarea value={dbConfig.sql} onChange={e=>setDbConfig({...dbConfig, sql:e.target.value})} className="w-full pl-4 pr-4 pt-7 pb-2 bg-zinc-50 border border-zinc-200 rounded-xl text-sm font-mono h-24 focus:bg-white resize-none" placeholder="SELECT * FROM table LIMIT 100"/>
                 </div>
                 <button onClick={()=>handleConnectDB(activeSession.id)} className="w-full py-3 bg-indigo-600 text-white rounded-xl font-semibold hover:bg-indigo-700 transition-all shadow-lg shadow-indigo-200">立即连接</button>
              </div>
           </div>
        </div>
      )}

      {/* Mode Modal - 模式与模型选择弹窗 */}
      {showModeModal && (
        <div className="absolute inset-0 z-[60] flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/20 backdrop-blur-sm" onClick={() => setShowModeModal(false)}/>
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-6 relative z-10 animate-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-zinc-900 flex items-center gap-2"><Command size={18} className="text-indigo-500"/> 模式与模型</h3>
              <button onClick={() => setShowModeModal(false)} className="p-1 rounded-full hover:bg-zinc-100 text-zinc-400 hover:text-zinc-900"><X size={18}/></button>
            </div>

            <div className="space-y-5">
              {/* 模式选择 */}
              <div>
                <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider block mb-2">运行模式</label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { k: 'ask', label: 'Ask', desc: '普通问答' },
                    { k: 'agent_single', label: 'Agent', desc: '单模型' },
                    { k: 'agent_multi', label: 'Agent+', desc: '多专家' },
                  ].map(opt => (
                    <button
                      key={opt.k}
                      onClick={() => updateSession(activeSession.id, { runMode: opt.k })}
                      className={`p-3 rounded-xl border-2 transition-all text-left ${String(activeSession.runMode || 'ask') === opt.k ? 'bg-indigo-50 border-indigo-500 text-indigo-900 shadow-sm' : 'bg-white border-zinc-200 text-zinc-700 hover:border-zinc-300'}`}
                    >
                      <div className="font-semibold text-sm">{opt.label}</div>
                      <div className="text-[10px] text-zinc-500 mt-0.5">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* 模型选择 */}
              <div>
                <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider block mb-2">模型选择</label>
                {activeSession.runMode !== 'agent_multi' ? (() => {
                  const modeKey = activeSession.runMode === 'agent_single' ? 'agent_single' : 'ask';
                  const ms = (activeSession.modelPrefs && activeSession.modelPrefs[modeKey]) || { provider: 'deepseekA', model: getDefaultModelFor('deepseekA', activeSession.runMode) };
                  const provider = ms.provider || 'deepseekA';
                  const opts = getProviderModelOptions(provider);
                  const model = ms.model && opts.includes(ms.model) ? ms.model : (getDefaultModelFor(provider, activeSession.runMode) || opts[0] || '');
                  return (
                    <div className="space-y-2">
                      <select
                        value={provider}
                        onChange={(e) => {
                          const p = e.target.value;
                          const nextModel = (getProviderModelOptions(p).includes(model) ? model : (getDefaultModelFor(p, activeSession.runMode) || getProviderModelOptions(p)[0] || ''));
                          updateSession(activeSession.id, { modelPrefs: { ...(activeSession.modelPrefs || {}), [modeKey]: { provider: p, model: nextModel } } });
                        }}
                        className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all"
                      >
                        {allProviderIds.map(p => <option key={p} value={p}>{PROVIDER_LABELS[p] || p}</option>)}
                      </select>
                      <select
                        value={model}
                        onChange={(e) => updateSession(activeSession.id, { modelPrefs: { ...(activeSession.modelPrefs || {}), [modeKey]: { provider, model: e.target.value } } })}
                        className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-3 py-2 text-sm focus:bg-white focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all"
                      >
                        {opts.map(m => <option key={m} value={m}>{m}</option>)}
                      </select>
                    </div>
                  );
                })() : (
              <div className="space-y-3">
                    {['planner','executor','verifier'].map(role => {
                      const r = (activeSession.modelPrefs?.agent_multi?.[role]) || { provider: role==='planner'?'deepseekA':role==='executor'?'deepseekB':'deepseekC', model: 'deepseek-reasoner' };
                      const p = r.provider;
                      const opts = getProviderModelOptions(p);
                      const m = r.model && opts.includes(r.model) ? r.model : (getDefaultModelFor(p, 'agent_single') || opts[0] || '');
                      const roleLabel = role === 'planner' ? '规划' : role === 'executor' ? '执行' : '评审';
                      return (
                        <div key={role} className="space-y-1">
                          <div className="text-[11px] font-bold text-zinc-600">{roleLabel}</div>
                          <div className="grid grid-cols-2 gap-2">
                            <select
                              value={p}
                              onChange={(e) => {
                                const np = e.target.value;
                                const nextModel = (getProviderModelOptions(np).includes(m) ? m : (getDefaultModelFor(np, 'agent_single') || getProviderModelOptions(np)[0] || ''));
                                updateSession(activeSession.id, {
                                  modelPrefs: {
                                    ...(activeSession.modelPrefs || {}),
                                    agent_multi: {
                                      ...(activeSession.modelPrefs?.agent_multi || {}),
                                      [role]: { provider: np, model: nextModel }
                                    }
                                  }
                                });
                              }}
                              className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-3 py-2 text-xs focus:bg-white focus:border-indigo-400 transition-all"
                            >
                              {allProviderIds.map(pid => <option key={pid} value={pid}>{PROVIDER_LABELS[pid] || pid}</option>)}
                            </select>
                            <select
                              value={m}
                              onChange={(e) => updateSession(activeSession.id, {
                                modelPrefs: {
                                  ...(activeSession.modelPrefs || {}),
                                  agent_multi: {
                                    ...(activeSession.modelPrefs?.agent_multi || {}),
                                    [role]: { provider: p, model: e.target.value }
                                  }
                                }
                              })}
                              className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-3 py-2 text-xs focus:bg-white focus:border-indigo-400 transition-all"
                            >
                              {opts.map(mm => <option key={mm} value={mm}>{mm}</option>)}
                            </select>
                 </div>
              </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* 视觉模型选择 */}
              <div>
                <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider block mb-2">视觉理解</label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={activeSession.visionEnabled !== false}
                      onChange={(e) => updateSession(activeSession.id, { visionEnabled: !!e.target.checked })}
                      className="rounded border-zinc-300 text-indigo-600 focus:ring-indigo-500"
                    />
                    <span className="text-sm text-zinc-700">启用视觉理解</span>
                  </label>
                  {activeSession.visionEnabled !== false && (
                    <div className="text-sm text-zinc-500 bg-zinc-50 border border-zinc-200 rounded-xl px-3 py-2">
                      视觉模型：Qwen-Omni-Turbo（固定使用）
                    </div>
                  )}
                </div>
              </div>

              <button 
                onClick={() => setShowModeModal(false)} 
                className="w-full py-2.5 bg-zinc-900 text-white rounded-xl font-semibold hover:bg-zinc-800 transition-all shadow-md shadow-zinc-200"
              >
                确定
              </button>
            </div>
           </div>
        </div>
      )}
    </div>
  );
}