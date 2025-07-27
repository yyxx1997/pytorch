#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Star 趋势 Web 仪表板

提供一个简单的 Web 界面来查看和分析 GitHub 仓库的 star 趋势。

作者：基于原项目扩展
许可证：MIT
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from star_tracker import GitHubStarTracker

# 页面配置
st.set_page_config(
    page_title="GitHub Star 趋势分析",
    page_icon="⭐",
    layout="wide"
)

# 侧边栏配置
st.sidebar.title("🔧 配置")
repo_input = st.sidebar.text_input(
    "仓库名称", 
    placeholder="例如: pytorch/pytorch",
    help="输入格式: owner/repo"
)

token_input = st.sidebar.text_input(
    "GitHub Token (可选)", 
    type="password",
    help="提供 token 可以增加 API 请求限制"
)

max_pages = st.sidebar.slider(
    "最大获取页数", 
    min_value=1, 
    max_value=20, 
    value=5,
    help="每页100个star，页数越多数据越完整但耗时更长"
)

# 主页面
st.title("📊 GitHub Star 趋势分析仪表板")
st.markdown("---")

if repo_input:
    try:
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 初始化追踪器
        status_text.text("正在初始化...")
        progress_bar.progress(10)
        tracker = GitHubStarTracker(token=token_input if token_input else None)
        
        # 获取仓库信息
        status_text.text("正在获取仓库信息...")
        progress_bar.progress(30)
        repo_info = tracker.get_repo_info(repo_input)
        
        # 获取 star 数据
        status_text.text("正在获取 star 数据...")
        progress_bar.progress(50)
        stars_data = tracker.get_stargazers_with_dates(repo_input, max_pages=max_pages)
        
        # 处理数据
        status_text.text("正在处理数据...")
        progress_bar.progress(80)
        df = tracker.process_star_data(stars_data)
        
        progress_bar.progress(100)
        status_text.text("数据加载完成！")
        
        # 隐藏进度条
        progress_bar.empty()
        status_text.empty()
        
        if df.empty:
            st.warning("⚠️ 未获取到足够的 star 数据，可能是因为：")
            st.markdown("""
            - 仓库star数量较少
            - API 请求限制
            - 仓库不存在或私有
            """)
        else:
            # 显示仓库基本信息
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⭐ Stars", repo_info.get('stargazers_count', 'N/A'))
            with col2:
                st.metric("🍴 Forks", repo_info.get('forks_count', 'N/A'))
            with col3:
                st.metric("👁️ Watchers", repo_info.get('watchers_count', 'N/A'))
            with col4:
                st.metric("📝 Issues", repo_info.get('open_issues_count', 'N/A'))
            
            # 仓库描述
            if repo_info.get('description'):
                st.info(f"📝 **描述**: {repo_info['description']}")
            
            # 语言和创建时间
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"🌐 **主要语言**: {repo_info.get('language', 'N/A')}")
            with col2:
                created_at = repo_info.get('created_at', 'N/A')
                if created_at != 'N/A':
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    st.write(f"📅 **创建时间**: {created_date.strftime('%Y-%m-%d')}")
            
            st.markdown("---")
            
            # 统计摘要
            if len(df) > 0:
                first_star = df['starred_at'].min()
                last_star = df['starred_at'].max()
                total_days = (last_star - first_star).days + 1
                avg_stars_per_day = len(df) / total_days if total_days > 0 else 0
                
                # 最近30天数据
                recent_30_days = datetime.now() - timedelta(days=30)
                recent_stars = df[df['starred_at'] >= recent_30_days]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📈 分析天数", f"{total_days}")
                with col2:
                    st.metric("📊 日均新增", f"{avg_stars_per_day:.2f}")
                with col3:
                    st.metric("🕒 近30天新增", len(recent_stars))
                with col4:
                    st.metric("📋 数据样本", len(df))
            
            # 可视化图表
            st.markdown("## 📈 趋势图表")
            
            # 累积趋势图
            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(
                x=df['starred_at'],
                y=df['cumulative_stars'],
                mode='lines',
                name='累积 Stars',
                line=dict(color='#FF6B6B', width=3),
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            
            fig_cumulative.update_layout(
                title=f'{repo_input} - Star 累积趋势',
                xaxis_title='时间',
                yaxis_title='累积 Star 数量',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_cumulative, use_container_width=True)
            
            # 每日新增趋势图
            daily_stars = df.groupby(df['starred_at'].dt.date).size().reset_index()
            daily_stars.columns = ['date', 'daily_count']
            
            fig_daily = px.bar(
                daily_stars, 
                x='date', 
                y='daily_count',
                title=f'{repo_input} - 每日新增 Star',
                color='daily_count',
                color_continuous_scale='Viridis'
            )
            
            fig_daily.update_layout(
                xaxis_title='时间',
                yaxis_title='每日新增 Star 数量',
                height=400
            )
            
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # 月度统计
            monthly_stats = df.groupby(df['starred_at'].dt.to_period('M')).size().reset_index()
            monthly_stats.columns = ['month', 'monthly_count']
            monthly_stats['month'] = monthly_stats['month'].astype(str)
            
            if len(monthly_stats) > 1:
                fig_monthly = px.line(
                    monthly_stats,
                    x='month',
                    y='monthly_count',
                    title=f'{repo_input} - 月度 Star 增长',
                    markers=True
                )
                
                fig_monthly.update_layout(
                    xaxis_title='月份',
                    yaxis_title='月度新增 Star 数量',
                    height=400
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            # 数据表格
            st.markdown("## 📋 详细数据")
            
            # 显示选项
            show_data = st.checkbox("显示原始数据")
            if show_data:
                st.dataframe(
                    df[['user', 'starred_at', 'cumulative_stars']].head(100),
                    height=300
                )
            
            # 导出功能
            st.markdown("## 💾 数据导出")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 下载 CSV 数据"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="下载 CSV 文件",
                        data=csv,
                        file_name=f"{repo_input.replace('/', '_')}_stars.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 下载统计报告"):
                    report = f"""
# {repo_input} Star 趋势报告

## 基本信息
- 仓库: {repo_input}
- 描述: {repo_info.get('description', '无')}
- 当前 Stars: {repo_info.get('stargazers_count', 'N/A')}
- Forks: {repo_info.get('forks_count', 'N/A')}
- 主要语言: {repo_info.get('language', 'N/A')}

## 趋势分析
- 分析样本: {len(df)} stars
- 分析天数: {total_days}
- 日均新增: {avg_stars_per_day:.2f}
- 近30天新增: {len(recent_stars)}

报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    st.download_button(
                        label="下载报告",
                        data=report,
                        file_name=f"{repo_input.replace('/', '_')}_report.md",
                        mime="text/markdown"
                    )
                    
    except Exception as e:
        st.error(f"❌ 发生错误: {str(e)}")
        st.markdown("""
        **可能的解决方案:**
        - 检查仓库名称格式是否正确 (owner/repo)
        - 确认仓库是公开的
        - 提供 GitHub Token 以增加 API 限制
        - 减少获取页数
        """)

else:
    # 欢迎页面
    st.markdown("""
    ## 👋 欢迎使用 GitHub Star 趋势分析工具
    
    这个工具可以帮助您：
    
    - 📊 **分析仓库的 star 增长趋势**
    - 📈 **查看每日/月度统计数据**
    - 💾 **导出数据和报告**
    - 🎯 **了解仓库的受欢迎程度变化**
    
    ### 使用方法：
    1. 在左侧边栏输入要分析的仓库名称 (格式: owner/repo)
    2. （可选）提供 GitHub Token 以获得更高的 API 限制
    3. 选择要获取的数据页数
    4. 点击分析按钮开始
    
    ### 示例仓库：
    - `pytorch/pytorch` - PyTorch 深度学习框架
    - `microsoft/vscode` - Visual Studio Code
    - `facebook/react` - React 前端框架
    - `tensorflow/tensorflow` - TensorFlow 机器学习框架
    
    ---
    💡 **提示**: 提供 GitHub Token 可以显著提高 API 请求限制，获得更完整的数据。
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌟 GitHub Star 趋势分析工具 | 基于 Streamlit 构建</p>
</div>
""", unsafe_allow_html=True)