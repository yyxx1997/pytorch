#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Star 趋势统计工具

该工具用于获取和分析 GitHub 仓库的 star 趋势数据，
并生成可视化图表来展示仓库的受欢迎程度变化。

作者：基于原项目扩展
许可证：MIT
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import argparse
import sys

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class GitHubStarTracker:
    """GitHub Star 趋势追踪器"""
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化追踪器
        
        Args:
            token: GitHub API token (可选，用于提高API限制)
        """
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3.star+json",
            "User-Agent": "Star-Trends-Tracker"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def get_repo_info(self, repo: str) -> Dict:
        """
        获取仓库基本信息
        
        Args:
            repo: 仓库名称，格式为 "owner/repo"
        
        Returns:
            包含仓库信息的字典
        """
        url = f"{self.base_url}/repos/{repo}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"无法获取仓库信息: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_stargazers_with_dates(self, repo: str, max_pages: int = 10) -> List[Dict]:
        """
        获取带有时间戳的 stargazer 数据
        
        Args:
            repo: 仓库名称
            max_pages: 最大获取页数
        
        Returns:
            stargazer 数据列表
        """
        all_stars = []
        page = 1
        
        print(f"正在获取 {repo} 的 star 数据...")
        
        while page <= max_pages:
            url = f"{self.base_url}/repos/{repo}/stargazers"
            params = {
                "page": page,
                "per_page": 100
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"请求失败: {response.status_code}")
                break
            
            stars = response.json()
            if not stars:
                break
            
            all_stars.extend(stars)
            print(f"已获取第 {page} 页，累计 {len(all_stars)} 个 star")
            page += 1
        
        return all_stars
    
    def process_star_data(self, stars_data: List[Dict]) -> pd.DataFrame:
        """
        处理 star 数据并转换为 DataFrame
        
        Args:
            stars_data: 原始 star 数据
        
        Returns:
            处理后的 DataFrame
        """
        processed_data = []
        
        for star in stars_data:
            if 'starred_at' in star:
                processed_data.append({
                    'user': star['user']['login'],
                    'starred_at': pd.to_datetime(star['starred_at']),
                    'user_id': star['user']['id']
                })
        
        df = pd.DataFrame(processed_data)
        if not df.empty:
            df = df.sort_values('starred_at')
            df['cumulative_stars'] = range(1, len(df) + 1)
        
        return df
    
    def generate_trend_chart(self, df: pd.DataFrame, repo: str, output_path: str = None):
        """
        生成 star 趋势图表
        
        Args:
            df: 处理后的数据
            repo: 仓库名称
            output_path: 输出路径（可选）
        """
        if df.empty:
            print("没有数据可以绘制")
            return
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. 累积 star 趋势图
        ax1.plot(df['starred_at'], df['cumulative_stars'], 
                linewidth=2, color='#FF6B6B', alpha=0.8)
        ax1.fill_between(df['starred_at'], df['cumulative_stars'], 
                        alpha=0.3, color='#FF6B6B')
        ax1.set_title(f'{repo} - Star 累积趋势', fontsize=16, fontweight='bold')
        ax1.set_xlabel('时间', fontsize=12)
        ax1.set_ylabel('累积 Star 数量', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. 每日新增 star 趋势图
        daily_stars = df.groupby(df['starred_at'].dt.date).size()
        daily_stars_df = pd.DataFrame({
            'date': daily_stars.index,
            'daily_stars': daily_stars.values
        })
        
        ax2.bar(daily_stars_df['date'], daily_stars_df['daily_stars'], 
               alpha=0.7, color='#4ECDC4', width=0.8)
        ax2.set_title(f'{repo} - 每日新增 Star', fontsize=16, fontweight='bold')
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('每日新增 Star 数量', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {output_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"star_trends_{repo.replace('/', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {filename}")
        
        plt.show()
    
    def generate_summary_report(self, df: pd.DataFrame, repo_info: Dict, repo: str):
        """
        生成摘要报告
        
        Args:
            df: 数据框
            repo_info: 仓库信息
            repo: 仓库名称
        """
        if df.empty:
            print("没有足够的数据生成报告")
            return
        
        print("\n" + "="*50)
        print(f"📊 {repo} Star 趋势分析报告")
        print("="*50)
        
        # 基本信息
        print(f"📍 仓库描述: {repo_info.get('description', '无描述')}")
        print(f"⭐ 当前 Star 数: {repo_info.get('stargazers_count', 'N/A')}")
        print(f"🍴 Fork 数: {repo_info.get('forks_count', 'N/A')}")
        print(f"👁️  Watch 数: {repo_info.get('watchers_count', 'N/A')}")
        print(f"🌐 主要语言: {repo_info.get('language', 'N/A')}")
        print(f"📅 创建时间: {repo_info.get('created_at', 'N/A')}")
        
        # 趋势分析
        if len(df) > 0:
            first_star = df['starred_at'].min()
            last_star = df['starred_at'].max()
            total_days = (last_star - first_star).days + 1
            avg_stars_per_day = len(df) / total_days if total_days > 0 else 0
            
            print(f"\n📈 趋势分析:")
            print(f"   首个 Star: {first_star.strftime('%Y-%m-%d')}")
            print(f"   最新 Star: {last_star.strftime('%Y-%m-%d')}")
            print(f"   分析天数: {total_days} 天")
            print(f"   平均每日新增: {avg_stars_per_day:.2f} stars")
            
            # 最近30天趋势
            recent_30_days = datetime.now() - timedelta(days=30)
            recent_stars = df[df['starred_at'] >= recent_30_days]
            if len(recent_stars) > 0:
                print(f"   最近30天新增: {len(recent_stars)} stars")
        
        print("="*50)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GitHub Star 趋势统计工具")
    parser.add_argument("repo", help="仓库名称，格式: owner/repo")
    parser.add_argument("--token", help="GitHub API Token (可选)")
    parser.add_argument("--output", help="输出图片路径 (可选)")
    parser.add_argument("--max-pages", type=int, default=10, 
                       help="最大获取页数 (默认: 10)")
    
    args = parser.parse_args()
    
    try:
        # 创建追踪器
        tracker = GitHubStarTracker(token=args.token)
        
        # 获取仓库信息
        repo_info = tracker.get_repo_info(args.repo)
        
        # 获取 star 数据
        stars_data = tracker.get_stargazers_with_dates(
            args.repo, max_pages=args.max_pages
        )
        
        if not stars_data:
            print("未获取到 star 数据")
            return
        
        # 处理数据
        df = tracker.process_star_data(stars_data)
        
        # 生成报告
        tracker.generate_summary_report(df, repo_info, args.repo)
        
        # 生成图表
        tracker.generate_trend_chart(df, args.repo, args.output)
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()