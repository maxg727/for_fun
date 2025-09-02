import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Optional, Dict, List, Tuple


class StandingsModule:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_standings_data(self, league_id: str) -> pd.DataFrame:
        """Get comprehensive standings data from database"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT 
                r.roster_id,
                r.owner_id,
                u.display_name,
                r.settings,
                r.players
            FROM rosters r
            LEFT JOIN users u ON r.owner_id = u.user_id
            WHERE r.league_id = ? AND r.owner_id IS NOT NULL
            ORDER BY r.roster_id
        """

        cursor = conn.cursor()
        cursor.execute(query, (league_id,))
        results = cursor.fetchall()
        conn.close()

        standings_data = []
        for row in results:
            roster_id, owner_id, display_name, settings_json, players_json = row

            # Parse settings JSON
            try:
                settings = json.loads(settings_json) if settings_json else {}
            except:
                settings = {}

            # Parse players JSON for roster size
            try:
                players = json.loads(players_json) if players_json else []
                roster_size = len([p for p in players if p])
            except:
                roster_size = 0

            wins = settings.get('wins', 0)
            losses = settings.get('losses', 0)
            ties = settings.get('ties', 0)
            points_for = settings.get('fpts', 0) or 0
            points_against = settings.get('fpts_against', 0) or 0

            total_games = wins + losses + ties
            win_pct = round(wins / max(1, total_games), 3) if total_games > 0 else 0.000

            standings_data.append({
                'Rank': 0,  # Will be calculated after sorting
                'Team': display_name or 'Unknown',
                'W': wins,
                'L': losses,
                'T': ties,
                'Win %': win_pct,
                'PF': round(points_for, 1),
                'PA': round(points_against, 1),
                'Diff': round(points_for - points_against, 1),
                'Avg PF': round(points_for / max(1, total_games), 1) if total_games > 0 else 0,
                'Avg PA': round(points_against / max(1, total_games), 1) if total_games > 0 else 0,
                'Roster Size': roster_size,
                'roster_id': roster_id,
                'owner_id': owner_id
            })

        df = pd.DataFrame(standings_data)

        if not df.empty:
            # Sort by wins (descending), then by points for (descending)
            df = df.sort_values(['W', 'PF'], ascending=[False, False])
            df = df.reset_index(drop=True)
            df['Rank'] = range(1, len(df) + 1)

        return df

    def get_weekly_scores(self, league_id: str) -> pd.DataFrame:
        """Get weekly scoring data for trends"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT 
                m.roster_id,
                m.week,
                m.points,
                u.display_name
            FROM matchups m
            LEFT JOIN rosters r ON m.roster_id = r.roster_id AND m.league_id = r.league_id
            LEFT JOIN users u ON r.owner_id = u.user_id
            WHERE m.league_id = ? AND m.points IS NOT NULL
            ORDER BY m.week, m.roster_id
        """

        df = pd.read_sql_query(query, conn, params=(league_id,))
        conn.close()

        return df

    def render_cumulative_scoring(self, weekly_df: pd.DataFrame, selected_teams: List[str]):
        """Render cumulative scoring chart"""
        if weekly_df.empty:
            return

        # Filter for selected teams and calculate cumulative scores
        cumulative_data = []
        for team in selected_teams:
            team_data = weekly_df[weekly_df['display_name'] == team].sort_values('week')
            if not team_data.empty:
                team_data['cumulative_points'] = team_data['points'].cumsum()
                cumulative_data.append(team_data)

        if not cumulative_data:
            return

        cumulative_df = pd.concat(cumulative_data, ignore_index=True)

        # Create cumulative scoring chart
        fig = px.line(
            cumulative_df,
            x='week',
            y='cumulative_points',
            color='display_name',
            title='Cumulative Points Scored',
            labels={'week': 'Week', 'cumulative_points': 'Cumulative Points', 'display_name': 'Team'},
            markers=True
        )

        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_standings_table(self, df: pd.DataFrame):
        """Render the main standings table"""
        if df.empty:
            st.warning("No standings data available")
            return

        st.subheader("📊 League Standings")

        # Display table with custom styling
        display_df = df[['Rank', 'Team', 'W', 'L', 'T', 'Win %', 'PF', 'PA', 'Diff', 'Avg PF']].copy()

        # Format the dataframe
        display_df = display_df.round({
            'Win %': 3,
            'PF': 1,
            'PA': 1,
            'Diff': 1,
            'Avg PF': 1
        })

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    def render_standings_charts(self, df: pd.DataFrame):
        """Render standings visualization charts"""
        if df.empty:
            return

        # Create subplot with 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Points Scored vs Win Percentage',
                'Points For vs Points Against',
                'Win Percentage by Team',
                'Point Differential'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Chart 1: Points vs Win %
        fig.add_trace(
            go.Scatter(
                x=df['PF'], y=df['Win %'],
                mode='markers+text',
                text=df['Team'],
                textposition='top center',
                marker=dict(size=12, color=df['W'], colorscale='RdYlGn', showscale=False),
                name='Teams',
                hovertemplate='<b>%{text}</b><br>Points For: %{x}<br>Win %: %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        # Chart 2: PF vs PA
        fig.add_trace(
            go.Scatter(
                x=df['PF'], y=df['PA'],
                mode='markers+text',
                text=df['Team'],
                textposition='top center',
                marker=dict(size=12, color=df['Win %'], colorscale='RdYlGn', showscale=False),
                name='Teams',
                hovertemplate='<b>%{text}</b><br>Points For: %{x}<br>Points Against: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # Chart 3: Win % by Team
        fig.add_trace(
            go.Bar(
                x=df['Team'], y=df['Win %'],
                marker_color=df['Win %'],
                marker_colorscale='RdYlGn',
                name='Win %',
                hovertemplate='<b>%{x}</b><br>Win Percentage: %{y:.1%}<extra></extra>'
            ),
            row=2, col=1
        )

        # Chart 4: Point Differential
        colors = ['green' if diff > 0 else 'red' for diff in df['Diff']]
        fig.add_trace(
            go.Bar(
                x=df['Team'], y=df['Diff'],
                marker_color=colors,
                name='Point Diff',
                hovertemplate='<b>%{x}</b><br>Point Differential: %{y:+.1f}<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="League Standings Analysis"
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Points For", row=1, col=1)
        fig.update_xaxes(title_text="Points For", row=1, col=2)
        fig.update_xaxes(title_text="Team", row=2, col=1, tickangle=45)
        fig.update_xaxes(title_text="Team", row=2, col=2, tickangle=45)

        # Update y-axis labels
        fig.update_yaxes(title_text="Win Percentage", row=1, col=1)
        fig.update_yaxes(title_text="Points Against", row=1, col=2)
        fig.update_yaxes(title_text="Win Percentage", row=2, col=1, tickformat='.1%')
        fig.update_yaxes(title_text="Point Differential", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

    def render_weekly_trends(self, league_id: str, standings_df: pd.DataFrame):
        """Render weekly scoring trends"""
        weekly_df = self.get_weekly_scores(league_id)

        if weekly_df.empty:
            st.info("No weekly scoring data available")
            return

        st.subheader("📈 Weekly Scoring Trends")

        # Select teams to show
        all_teams = weekly_df['display_name'].dropna().unique()
        if len(all_teams) > 8:  # Limit for readability
            # Show top 4 and bottom 4 teams by current standings
            if not standings_df.empty:
                top_teams = standings_df.head(4)['Team'].tolist()
                bottom_teams = standings_df.tail(4)['Team'].tolist()
                selected_teams = top_teams + bottom_teams
            else:
                selected_teams = all_teams[:8]
        else:
            selected_teams = all_teams

        # Filter data
        trend_data = weekly_df[weekly_df['display_name'].isin(selected_teams)]

        # Create line chart
        fig = px.line(
            trend_data,
            x='week',
            y='points',
            color='display_name',
            title='Weekly Points Scored',
            labels={'week': 'Week', 'points': 'Points', 'display_name': 'Team'},
            markers=True
        )

        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add cumulative chart right here
        self.render_cumulative_scoring(weekly_df, selected_teams)

        # Weekly stats summary
        col1, col2, col3 = st.columns(3)

        with col1:
            if not weekly_df.empty:
                highest_week = weekly_df.loc[weekly_df['points'].idxmax()]
                st.metric(
                    "Highest Weekly Score",
                    f"{highest_week['points']:.1f}",
                    f"{highest_week['display_name']} (Week {highest_week['week']})"
                )

        with col2:
            if not weekly_df.empty:
                avg_score = weekly_df['points'].mean()
                st.metric("League Average", f"{avg_score:.1f}")

        with col3:
            if not weekly_df.empty:
                weeks_played = weekly_df['week'].nunique()
                st.metric("Weeks Completed", weeks_played)

    def render_power_rankings(self, df: pd.DataFrame):
        """Render power rankings based on recent performance"""
        if df.empty:
            return

        st.subheader("⚡ Power Rankings")
        st.caption("Based on points scored and win percentage")

        try:
            # Calculate power score (combination of win % and scoring)
            df_power = df.copy()

            # Check for division by zero issues
            if df_power['Win %'].max() == df_power['Win %'].min():
                win_pct_norm = pd.Series([0.5] * len(df_power))
            else:
                win_pct_norm = (df_power['Win %'] - df_power['Win %'].min()) / (
                            df_power['Win %'].max() - df_power['Win %'].min())

            if df_power['PF'].max() == df_power['PF'].min():
                pf_norm = pd.Series([0.5] * len(df_power))
            else:
                pf_norm = (df_power['PF'] - df_power['PF'].min()) / (df_power['PF'].max() - df_power['PF'].min())

            df_power['win_pct_norm'] = win_pct_norm.values
            df_power['pf_norm'] = pf_norm.values

            # Power score: 60% win percentage, 40% points scored
            df_power['power_score'] = (df_power['win_pct_norm'] * 0.6) + (df_power['pf_norm'] * 0.4)
            df_power = df_power.sort_values('power_score', ascending=False).reset_index(drop=True)
            df_power['power_rank'] = range(1, len(df_power) + 1)

            # Display power rankings
            power_display = df_power[['power_rank', 'Team', 'W', 'L', 'Win %', 'PF', 'power_score']].copy()
            power_display.columns = ['Power Rank', 'Team', 'W', 'L', 'Win %', 'PF', 'Power Score']
            power_display = power_display.round({
                'Win %': 3,
                'PF': 1,
                'Power Score': 3
            })

            st.dataframe(power_display, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error in power rankings: {str(e)}")

    def render_team_momentum(self, league_id: str, standings_df: pd.DataFrame):
        """Render teams that are heating up or cooling down"""
        if standings_df.empty:
            return

        try:
            # Get all weekly scores
            conn = sqlite3.connect(self.db_path)

            # Get the latest week with data
            latest_week_query = "SELECT MAX(week) FROM matchups WHERE league_id = ? AND points IS NOT NULL"
            cursor = conn.cursor()
            cursor.execute(latest_week_query, (league_id,))
            result = cursor.fetchone()
            latest_week = result[0] if result and result[0] else 1

            # Get all weekly data
            all_scores_query = """
                SELECT 
                    m.roster_id,
                    m.week,
                    m.points,
                    u.display_name
                FROM matchups m
                LEFT JOIN rosters r ON m.roster_id = r.roster_id AND m.league_id = r.league_id
                LEFT JOIN users u ON r.owner_id = u.user_id
                WHERE m.league_id = ? 
                AND m.points IS NOT NULL
                ORDER BY m.roster_id, m.week
            """

            all_scores_df = pd.read_sql_query(all_scores_query, conn, params=(league_id,))
            conn.close()

            if all_scores_df.empty or latest_week < 4:
                st.info("Not enough data to calculate momentum (need at least 4 weeks)")
                return

            st.subheader("🔥❄️ Team Momentum")

            # Define recent period (last 3 weeks) and previous period
            recent_weeks = 3
            recent_start = max(1, latest_week - recent_weeks + 1)

            st.caption(f"Recent form (weeks {recent_start}-{latest_week}) vs Previous form")

            # Calculate momentum for each team
            momentum_data = []

            for team_name in standings_df['Team'].values:
                team_scores = all_scores_df[all_scores_df['display_name'] == team_name]

                if team_scores.empty:
                    continue

                # Split into recent and previous periods
                recent_scores = team_scores[team_scores['week'] >= recent_start]['points'].tolist()
                previous_scores = team_scores[team_scores['week'] < recent_start]['points'].tolist()

                # Need data from both periods
                if len(recent_scores) >= 2 and len(previous_scores) >= 1:
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    previous_avg = sum(previous_scores) / len(previous_scores)

                    # Calculate momentum
                    momentum = recent_avg - previous_avg
                    momentum_pct = (momentum / previous_avg) * 100 if previous_avg > 0 else 0

                    momentum_data.append({
                        'Team': team_name,
                        'Recent Avg': round(recent_avg, 1),
                        'Previous Avg': round(previous_avg, 1),
                        'Momentum': round(momentum, 1),
                        'Momentum %': round(momentum_pct, 1)
                    })

            if not momentum_data:
                st.info("Unable to calculate momentum - insufficient data")
                return

            momentum_df = pd.DataFrame(momentum_data)
            momentum_df = momentum_df.sort_values('Momentum', ascending=True)  # Sort for visualization

            # Create momentum visualization
            fig = go.Figure()

            # Determine colors based on momentum value
            colors = ['green' if m > 0 else 'red' for m in momentum_df['Momentum']]

            # Add scatter plot on single axis
            fig.add_trace(go.Scatter(
                x=momentum_df['Momentum'],
                y=[0] * len(momentum_df),  # All on same y-axis level
                mode='markers+text',
                text=momentum_df['Team'],
                textposition='top center',
                textfont=dict(size=10),
                marker=dict(
                    size=15,
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                              'Momentum: %{x:+.1f} pts<br>' +
                              'Recent Avg: %{customdata[0]:.1f}<br>' +
                              'Previous Avg: %{customdata[1]:.1f}<br>' +
                              'Change: %{customdata[2]:+.1f}%<extra></extra>',
                customdata=momentum_df[['Recent Avg', 'Previous Avg', 'Momentum %']].values,
                showlegend=False
            ))

            # Add vertical line at zero
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

            # Add momentum zones
            max_momentum = momentum_df['Momentum'].max()
            min_momentum = momentum_df['Momentum'].min()

            fig.add_annotation(
                x=max_momentum * 0.7,
                y=0.15,
                text="🔥 HEATING UP",
                showarrow=False,
                font=dict(size=14, color="green"),
                bgcolor="rgba(0,255,0,0.1)"
            )

            fig.add_annotation(
                x=min_momentum * 0.7,
                y=0.15,
                text="❄️ COOLING DOWN",
                showarrow=False,
                font=dict(size=14, color="red"),
                bgcolor="rgba(255,0,0,0.1)"
            )

            fig.update_layout(
                title='Team Momentum Tracker',
                xaxis_title='Momentum (Points Change from Previous Average)',
                height=400,
                showlegend=False,
                hovermode='closest',
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-0.3, 0.3]
                ),
                xaxis=dict(
                    showgrid=True,
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='gray'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display summary stats
            col1, col2, col3 = st.columns(3)

            with col1:
                biggest_riser = momentum_df.iloc[-1]  # Last one (highest momentum)
                st.metric(
                    "🔥 Hottest Team",
                    biggest_riser['Team'],
                    f"+{biggest_riser['Momentum']:.1f} pts"
                )

            with col2:
                biggest_faller = momentum_df.iloc[0]  # First one (lowest momentum)
                st.metric(
                    "❄️ Coldest Team",
                    biggest_faller['Team'],
                    f"{biggest_faller['Momentum']:.1f} pts"
                )

            with col3:
                stable_teams = momentum_df[abs(momentum_df['Momentum']) < 5]
                st.metric(
                    "😐 Stable Teams",
                    len(stable_teams),
                    "< ±5 pts change"
                )

        except Exception as e:
            st.error(f"Error calculating momentum: {str(e)}")

    def render_consistency_analysis(self, league_id: str, standings_df: pd.DataFrame):
        """Render consistency vs scoring analysis"""
        if standings_df.empty:
            return

        try:
            # Get weekly scores for consistency calculation
            weekly_df = self.get_weekly_scores(league_id)

            if weekly_df.empty:
                st.info("No weekly scoring data available for consistency analysis")
                return

            st.subheader("🎯 Consistency Analysis")
            st.caption("Teams categorized by scoring consistency and average points")

            # Calculate consistency metrics for each team
            consistency_data = []

            for team_name in standings_df['Team'].values:
                team_scores = weekly_df[weekly_df['display_name'] == team_name]['points']

                if len(team_scores) >= 3:  # Need at least 3 games
                    avg_score = team_scores.mean()
                    std_dev = team_scores.std()
                    # Consistency score: inverse of coefficient of variation (lower CV = more consistent)
                    # We invert it so higher values mean more consistent
                    consistency_score = (avg_score / std_dev) if std_dev > 0 else 10

                    consistency_data.append({
                        'Team': team_name,
                        'Average Score': round(avg_score, 1),
                        'Std Dev': round(std_dev, 1),
                        'Consistency': round(consistency_score, 2),
                        'High Score': round(team_scores.max(), 1),
                        'Low Score': round(team_scores.min(), 1),
                        'Range': round(team_scores.max() - team_scores.min(), 1)
                    })

            if not consistency_data:
                st.info("Unable to calculate consistency - insufficient data")
                return

            consistency_df = pd.DataFrame(consistency_data)

            # Calculate median values for quadrant lines
            median_score = consistency_df['Average Score'].median()
            median_consistency = consistency_df['Consistency'].median()

            # Create scatter plot with quadrants
            fig = go.Figure()

            # Determine quadrant for each team
            quadrant_colors = []
            quadrant_labels = []

            for _, row in consistency_df.iterrows():
                score = row['Average Score']
                consistency = row['Consistency']

                if score >= median_score and consistency >= median_consistency:
                    color = 'green'
                    label = 'Elite (High Score, High Consistency)'
                elif score >= median_score and consistency < median_consistency:
                    color = 'orange'
                    label = 'Boom/Bust (High Score, Low Consistency)'
                elif score < median_score and consistency >= median_consistency:
                    color = 'blue'
                    label = 'Steady (Low Score, High Consistency)'
                else:
                    color = 'red'
                    label = 'Struggling (Low Score, Low Consistency)'

                quadrant_colors.append(color)
                quadrant_labels.append(label)

            # Add scatter points
            fig.add_trace(go.Scatter(
                x=consistency_df['Consistency'],
                y=consistency_df['Average Score'],
                mode='markers+text',
                text=consistency_df['Team'],
                textposition='top center',
                textfont=dict(size=10),
                marker=dict(
                    size=15,
                    color=quadrant_colors,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                              'Average Score: %{y:.1f}<br>' +
                              'Consistency Score: %{x:.2f}<br>' +
                              'Std Dev: %{customdata[0]:.1f}<br>' +
                              'Range: %{customdata[1]:.1f} pts<br>' +
                              'High: %{customdata[2]:.1f}<br>' +
                              'Low: %{customdata[3]:.1f}<extra></extra>',
                customdata=consistency_df[['Std Dev', 'Range', 'High Score', 'Low Score']].values,
                showlegend=False
            ))

            # Add quadrant lines
            fig.add_vline(x=median_consistency, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=median_score, line_dash="dash", line_color="gray", opacity=0.5)

            # Add quadrant labels
            max_consistency = consistency_df['Consistency'].max()
            min_consistency = consistency_df['Consistency'].min()
            max_score = consistency_df['Average Score'].max()
            min_score = consistency_df['Average Score'].min()

            # Quadrant annotations
            fig.add_annotation(
                x=median_consistency + (max_consistency - median_consistency) * 0.5,
                y=median_score + (max_score - median_score) * 0.5,
                text="🏆 ELITE<br>Consistent & High-Scoring",
                showarrow=False,
                font=dict(size=11, color="green"),
                bgcolor="rgba(0,255,0,0.1)",
                bordercolor="green",
                borderwidth=1
            )

            fig.add_annotation(
                x=min_consistency + (median_consistency - min_consistency) * 0.5,
                y=median_score + (max_score - median_score) * 0.5,
                text="💥 BOOM/BUST<br>Volatile & High-Scoring",
                showarrow=False,
                font=dict(size=11, color="orange"),
                bgcolor="rgba(255,165,0,0.1)",
                bordercolor="orange",
                borderwidth=1
            )

            fig.add_annotation(
                x=median_consistency + (max_consistency - median_consistency) * 0.5,
                y=min_score + (median_score - min_score) * 0.5,
                text="🛡️ STEADY<br>Consistent & Low-Scoring",
                showarrow=False,
                font=dict(size=11, color="blue"),
                bgcolor="rgba(0,0,255,0.1)",
                bordercolor="blue",
                borderwidth=1
            )

            fig.add_annotation(
                x=min_consistency + (median_consistency - min_consistency) * 0.5,
                y=min_score + (median_score - min_score) * 0.5,
                text="⚠️ STRUGGLING<br>Volatile & Low-Scoring",
                showarrow=False,
                font=dict(size=11, color="red"),
                bgcolor="rgba(255,0,0,0.1)",
                bordercolor="red",
                borderwidth=1
            )

            fig.update_layout(
                title='Team Consistency vs Scoring Analysis',
                xaxis_title='Consistency Score (Higher = More Consistent)',
                yaxis_title='Average Points Scored',
                height=600,
                showlegend=False,
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display quadrant summaries
            st.markdown("### 📊 Quadrant Breakdown")

            col1, col2 = st.columns(2)

            with col1:
                # Elite teams
                elite_teams = consistency_df[
                    (consistency_df['Average Score'] >= median_score) &
                    (consistency_df['Consistency'] >= median_consistency)
                    ].sort_values('Average Score', ascending=False)

                st.markdown("#### 🏆 Elite Teams")
                if not elite_teams.empty:
                    for _, team in elite_teams.iterrows():
                        st.text(f"• {team['Team']}: {team['Average Score']:.1f} pts (±{team['Std Dev']:.1f})")
                else:
                    st.info("No teams in elite quadrant")

                # Boom/Bust teams
                boom_bust = consistency_df[
                    (consistency_df['Average Score'] >= median_score) &
                    (consistency_df['Consistency'] < median_consistency)
                    ].sort_values('Average Score', ascending=False)

                st.markdown("#### 💥 Boom/Bust Teams")
                if not boom_bust.empty:
                    for _, team in boom_bust.iterrows():
                        st.text(f"• {team['Team']}: {team['Average Score']:.1f} pts (±{team['Std Dev']:.1f})")
                else:
                    st.info("No teams in boom/bust quadrant")

            with col2:
                # Steady teams
                steady_teams = consistency_df[
                    (consistency_df['Average Score'] < median_score) &
                    (consistency_df['Consistency'] >= median_consistency)
                    ].sort_values('Consistency', ascending=False)

                st.markdown("#### 🛡️ Steady Teams")
                if not steady_teams.empty:
                    for _, team in steady_teams.iterrows():
                        st.text(f"• {team['Team']}: {team['Average Score']:.1f} pts (±{team['Std Dev']:.1f})")
                else:
                    st.info("No teams in steady quadrant")

                # Struggling teams
                struggling = consistency_df[
                    (consistency_df['Average Score'] < median_score) &
                    (consistency_df['Consistency'] < median_consistency)
                    ].sort_values('Average Score', ascending=True)

                st.markdown("#### ⚠️ Struggling Teams")
                if not struggling.empty:
                    for _, team in struggling.iterrows():
                        st.text(f"• {team['Team']}: {team['Average Score']:.1f} pts (±{team['Std Dev']:.1f})")
                else:
                    st.info("No teams in struggling quadrant")

        except Exception as e:
            st.error(f"Error in consistency analysis: {str(e)}")

    def render(self, league_id: str, season: str):
        """Main render function for standings module"""
        st.title("📊 League Standings")

        # Get standings data
        standings_df = self.get_standings_data(league_id)

        if standings_df.empty:
            st.error("No standings data found for this league.")
            return

        # Main standings table
        self.render_standings_table(standings_df)

        # Visualizations
        st.markdown("---")
        self.render_standings_charts(standings_df)

        # Weekly trends
        st.markdown("---")
        self.render_weekly_trends(league_id, standings_df)

        # Power rankings
        st.markdown("---")
        self.render_power_rankings(standings_df)

        # Team momentum
        st.markdown("---")
        self.render_team_momentum(league_id, standings_df)

        # Consistency analysis
        st.markdown("---")
        self.render_consistency_analysis(league_id, standings_df)