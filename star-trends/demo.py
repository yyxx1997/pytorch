#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Star 趋势分析工具演示脚本

这个脚本演示了如何使用 GitHubStarTracker 类进行基本的分析。

使用方法:
    python demo.py

注意: 
    - 确保已安装所有依赖: pip install -r requirements.txt
    - 建议设置 GITHUB_TOKEN 环境变量以获得更好的 API 限制
"""

import os
from star_tracker import GitHubStarTracker

def demo_analysis():
    """演示基本的分析功能"""
    
    # 从环境变量获取 token (可选)
    token = os.getenv('GITHUB_TOKEN')
    
    # 创建分析器
    tracker = GitHubStarTracker(token=token)
    
    # 演示仓库列表 (选择star数量适中的仓库以便快速演示)
    demo_repos = [
        "streamlit/streamlit",  # 流行的 Web 应用框架
        "gradio-app/gradio",    # 机器学习演示工具
        "plotly/plotly.py",     # 数据可视化库
    ]
    
    print("🌟 GitHub Star 趋势分析演示")
    print("="*50)
    
    for i, repo in enumerate(demo_repos, 1):
        print(f"\n📊 演示 {i}/{len(demo_repos)}: 分析 {repo}")
        print("-" * 40)
        
        try:
            # 获取仓库信息
            print("📋 获取仓库基本信息...")
            repo_info = tracker.get_repo_info(repo)
            
            # 显示基本信息
            print(f"   📝 描述: {repo_info.get('description', '无描述')[:60]}...")
            print(f"   ⭐ 当前 Stars: {repo_info.get('stargazers_count', 'N/A'):,}")
            print(f"   🍴 Forks: {repo_info.get('forks_count', 'N/A'):,}")
            print(f"   🌐 语言: {repo_info.get('language', 'N/A')}")
            
            # 获取 star 数据 (限制为2页以便快速演示)
            print("📈 获取 star 趋势数据...")
            stars_data = tracker.get_stargazers_with_dates(repo, max_pages=2)
            
            if stars_data:
                # 处理数据
                df = tracker.process_star_data(stars_data)
                
                if not df.empty:
                    # 生成简要报告
                    first_star = df['starred_at'].min()
                    last_star = df['starred_at'].max()
                    total_days = (last_star - first_star).days + 1
                    avg_per_day = len(df) / total_days if total_days > 0 else 0
                    
                    print(f"   📊 分析了 {len(df)} 个 star (最近 {total_days} 天)")
                    print(f"   📈 平均每日新增: {avg_per_day:.2f} stars")
                    print(f"   🕒 数据范围: {first_star.strftime('%Y-%m-%d')} 到 {last_star.strftime('%Y-%m-%d')}")
                    
                    # 生成图表 (可选)
                    output_file = f"demo_{repo.replace('/', '_')}.png"
                    print(f"   🎨 生成趋势图: {output_file}")
                    tracker.generate_trend_chart(df, repo, output_file)
                else:
                    print("   ⚠️  没有获取到有效的 star 数据")
            else:
                print("   ⚠️  无法获取 star 数据")
                
        except Exception as e:
            print(f"   ❌ 分析 {repo} 时出错: {e}")
        
        # 分隔线
        if i < len(demo_repos):
            print("\n" + "="*50)
    
    print("\n🎉 演示完成!")
    print("\n💡 提示:")
    print("   - 使用完整的 star_tracker.py 脚本获取更详细的分析")
    print("   - 运行 'streamlit run web_dashboard.py' 启动 Web 界面")
    print("   - 设置 GITHUB_TOKEN 环境变量以获得更好的 API 限制")

def quick_demo():
    """快速演示单个仓库分析"""
    
    print("🚀 快速演示模式")
    print("="*30)
    
    # 使用一个小型但活跃的仓库进行演示
    demo_repo = "streamlit/streamlit"
    
    token = os.getenv('GITHUB_TOKEN')
    tracker = GitHubStarTracker(token=token)
    
    try:
        print(f"📊 分析仓库: {demo_repo}")
        
        # 获取仓库信息
        repo_info = tracker.get_repo_info(demo_repo)
        
        # 获取少量 star 数据
        stars_data = tracker.get_stargazers_with_dates(demo_repo, max_pages=1)
        
        # 处理数据
        df = tracker.process_star_data(stars_data)
        
        # 生成报告
        tracker.generate_summary_report(df, repo_info, demo_repo)
        
        print("✅ 快速演示完成!")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("💡 可能的解决方案:")
        print("   - 检查网络连接")
        print("   - 设置 GITHUB_TOKEN 环境变量")
        print("   - 稍后重试")

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_demo()
    else:
        demo_analysis()