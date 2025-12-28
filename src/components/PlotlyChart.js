import React, { useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';

/**
 * PlotlyChart 组件 - 渲染交互式 Plotly 图表
 * 
 * @param {Object} props
 * @param {string} props.plotlyJson - Plotly Figure 的 JSON 字符串
 * @param {string} props.className - 额外的 CSS 类名
 */
const PlotlyChart = ({ plotlyJson, className = '' }) => {
  const containerRef = useRef(null);

  if (!plotlyJson) {
    return null;
  }

  let plotData;
  try {
    plotData = typeof plotlyJson === 'string' ? JSON.parse(plotlyJson) : plotlyJson;
  } catch (error) {
    console.error('Failed to parse Plotly JSON:', error);
    return (
      <div className={`p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm ${className}`}>
        图表数据解析失败
      </div>
    );
  }

  // 默认配置
  const layout = {
    ...plotData.layout,
    autosize: true,
    margin: { l: 50, r: 50, t: 50, b: 50 },
    font: {
      family: 'system-ui, -apple-system, sans-serif',
      size: 12,
      color: '#374151'
    },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    // 支持中文
    ...(plotData.layout || {})
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    responsive: true,
    toImageButtonOptions: {
      format: 'png',
      filename: 'radarm-chart',
      height: 600,
      width: 800,
      scale: 2
    }
  };

  return (
    <div 
      ref={containerRef}
      className={`mt-2 w-full bg-white border border-zinc-200 rounded-lg shadow-sm overflow-hidden ${className}`}
      style={{ minHeight: '300px' }}
    >
      <Plot
        data={plotData.data || []}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%', minHeight: '300px' }}
        useResizeHandler={true}
      />
    </div>
  );
};

export default PlotlyChart;

