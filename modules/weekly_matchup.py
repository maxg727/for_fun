import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from functools import lru_cache


@lru_cache(maxsize=128)
def _safe_json_loads(blob: Optional[str], default_type: str = "dict"):
    """Cache-enabled JSON parsing with type-specific defaults."""
    if blob is None or blob == "":
        return {} if default_type == "dict" else []
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return {} if default_type == "dict" else []


class WeeklyMatchupModule:
    """
    Clean Weekly Matchup Analysis module for Sleeper leagues.

    Features:
    - Actual vs Expected vs Optimal scoring
    - Over/Under performers identification
    - Head-to-head player comparisons
    - Bench analysis
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection = None

    @property
    def connection(self):
        """Get database connection, creating one if needed."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
        return self._connection

    @st.cache_data(ttl=300)
    def get_available_weeks(_self, league_id: str) -> List[int]:
        """Get list of weeks with matchup data."""
        query = """
            SELECT DISTINCT week 
            FROM matchups 
            WHERE league_id = ? AND points IS NOT NULL 
            ORDER BY week
        """
        with _self.connection as conn:
            df = pd.read_sql(query, conn, params=(league_id,))
        return df['week'].tolist()

    @st.cache_data(ttl=300)
    def get_week_matchups(_self, league_id: str, week: int) -> pd.DataFrame:
        """Get all matchups for a specific week with opponent data."""
        query = """
            WITH week_matchups AS (
                SELECT 
                    m.roster_id,
                    m.matchup_id,
                    m.points,
                    m.starters,
                    m.starters_points,
                    m.players_points,
                    COALESCE(u.display_name, 'Team ' || m.roster_id) as team_name
                FROM matchups m
                LEFT JOIN rosters r ON m.roster_id = r.roster_id AND m.league_id = r.league_id
                LEFT JOIN users u ON r.owner_id = u.user_id
                WHERE m.league_id = ? AND m.week = ?
            )
            SELECT 
                m1.roster_id,
                m1.team_name,
                m1.points,
                m1.starters,
                m1.starters_points,
                m1.players_points,
                m1.matchup_id,
                m2.roster_id as opponent_roster_id,
                m2.team_name as opponent_name,
                m2.points as opponent_points,
                m2.starters as opponent_starters,
                m2.players_points as opponent_players_points
            FROM week_matchups m1
            LEFT JOIN week_matchups m2 
                ON m1.matchup_id = m2.matchup_id 
                AND m1.roster_id != m2.roster_id
            WHERE m1.points IS NOT NULL
            ORDER BY m1.matchup_id, m1.roster_id
        """

        with _self.connection as conn:
            df = pd.read_sql(query, conn, params=(league_id, week))

        # Parse JSON columns
        json_columns = {
            'starters': 'list',
            'starters_points': 'list',
            'players_points': 'dict',
            'opponent_starters': 'list',
            'opponent_players_points': 'dict'
        }

        for col, default_type in json_columns.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: _safe_json_loads(x, default_type))

        return df

    @st.cache_data(ttl=3600)
    def get_players_table(_self) -> pd.DataFrame:
        """Load player metadata."""
        query = """
            SELECT DISTINCT player_id, 
                   COALESCE(full_name, 'Unknown') as full_name, 
                   COALESCE(position, 'UNK') as position 
            FROM players 
            WHERE player_id IS NOT NULL
        """
        with _self.connection as conn:
            return pd.read_sql(query, conn)

    @st.cache_data(ttl=300)
    def get_league_week_scores(_self, league_id: str, week: int) -> pd.DataFrame:
        """Get all team scores for a specific week."""
        query = """
            SELECT roster_id, points
            FROM matchups 
            WHERE league_id = ? AND week = ? AND points IS NOT NULL
        """
        with _self.connection as conn:
            return pd.read_sql(query, conn, params=(league_id, week))

    @st.cache_data(ttl=300)
    def get_week_position_averages(_self, league_id: str, week: int) -> Dict[str, float]:
        """Calculate average points by position for starters in a specific week."""
        query = """
            SELECT starters, players_points
            FROM matchups 
            WHERE league_id = ? AND week = ? AND points IS NOT NULL
        """
        with _self.connection as conn:
            df = pd.read_sql(query, conn, params=(league_id, week))

        # Get players data
        players_df = _self.get_players_table()
        pos_map = dict(zip(players_df['player_id'], players_df['position']))

        position_points = {}
        position_counts = {}

        for _, row in df.iterrows():
            starters = _safe_json_loads(row['starters'], 'list')
            players_points = _safe_json_loads(row['players_points'], 'dict')

            for player_id in starters:
                if player_id in players_points:
                    position = pos_map.get(player_id, 'UNK')
                    points = float(players_points[player_id])

                    if position not in position_points:
                        position_points[position] = 0
                        position_counts[position] = 0

                    position_points[position] += points
                    position_counts[position] += 1

        # Calculate averages
        position_averages = {}
        for pos in position_points:
            if position_counts[pos] > 0:
                position_averages[pos] = position_points[pos] / position_counts[pos]

        return position_averages

    def calculate_matchup_analysis(self, matchup_data: pd.Series, opponent_data: pd.Series,
                                   league_scores: pd.DataFrame, players_df: pd.DataFrame,
                                   position_averages: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive matchup analysis."""
        analysis = {}

        # Basic scores
        actual_score = float(matchup_data.get('points', 0))
        opponent_score = float(opponent_data.get('points', 0)) if not pd.isna(opponent_data.get('points')) else 0

        # Win/Loss determination
        if actual_score > opponent_score:
            result = 'W'
        elif actual_score < opponent_score:
            result = 'L'
        else:
            result = 'T'

        analysis['result'] = result
        analysis['actual_score'] = actual_score
        analysis['opponent_score'] = opponent_score
        analysis['margin'] = actual_score - opponent_score

        # Optimal score calculation
        players_points = matchup_data.get('players_points', {}) or {}
        starters = matchup_data.get('starters', []) or []

        if players_points and starters:
            # Optimal lineup: top N scorers where N = number of starters
            all_scores = sorted(players_points.values(), reverse=True)
            optimal_score = sum(all_scores[:len(starters)])
            analysis['optimal_score'] = optimal_score
            analysis['lineup_efficiency'] = actual_score / optimal_score if optimal_score > 0 else 1.0
            analysis['points_left_on_bench'] = optimal_score - actual_score
        else:
            analysis['optimal_score'] = actual_score
            analysis['lineup_efficiency'] = 1.0
            analysis['points_left_on_bench'] = 0.0

        # Expected score (league median for the week)
        if not league_scores.empty:
            expected_score = league_scores['points'].median()
            analysis['expected_score'] = expected_score
            analysis['vs_expected'] = actual_score - expected_score

            # Calculate expected win % - what % of league would this score beat
            teams_beaten = (league_scores['points'] < actual_score).sum()
            total_teams = len(league_scores)
            analysis['expected_win_pct'] = (teams_beaten / (total_teams-1)) if total_teams > 0 else 0.0
        else:
            analysis['expected_score'] = actual_score
            analysis['vs_expected'] = 0.0
            analysis['expected_win_pct'] = 0.5  # Default to 50% if no data

        # Player performance comparison
        analysis['player_comparison'] = self._get_player_performance_comparison(
            players_points, starters, players_df, position_averages)

        return analysis

    def _get_player_performance_comparison(self, players_points: Dict, starters: List,
                                           players_df: pd.DataFrame,
                                           position_averages: Dict[str, float]) -> pd.DataFrame:
        """Get all players' actual vs position average performance."""
        if not players_points:
            return pd.DataFrame()

        name_map = dict(zip(players_df['player_id'], players_df['full_name']))
        pos_map = dict(zip(players_df['player_id'], players_df['position']))

        performance_data = []
        for pid, actual_pts in players_points.items():
            position = pos_map.get(pid, 'UNK')
            expected_pts = position_averages.get(position, 0.0)
            diff = actual_pts - expected_pts

            performance_data.append({
                'player_id': pid,
                'name': name_map.get(pid, 'Unknown'),
                'position': position,
                'actual_points': round(actual_pts, 1),
                'position_avg': round(expected_pts, 1),
                'difference': round(diff, 1),
                'is_starter': pid in starters,
                'performance_pct': round((actual_pts / expected_pts * 100) if expected_pts > 0 else 100, 1)
            })

        df = pd.DataFrame(performance_data)
        return df.sort_values('difference', ascending=False)

    def render_matchup_selector(self, league_id: str) -> Tuple[int, int]:
        """Render week and matchup selector."""
        st.title("Weekly Matchup Analysis")

        available_weeks = self.get_available_weeks(league_id)

        if not available_weeks:
            st.error("No matchup data available.")
            return None, None

        col1, col2 = st.columns(2)

        with col1:
            selected_week = st.selectbox(
                "Select Week",
                options=available_weeks,
                index=len(available_weeks) - 1
            )

        # Get matchups for selected week
        week_matchups = self.get_week_matchups(league_id, selected_week)

        if week_matchups.empty:
            st.error(f"No matchup data for week {selected_week}")
            return selected_week, None

        # Create matchup options
        matchup_options = []
        seen_matchups = set()

        for _, row in week_matchups.iterrows():
            matchup_id = row['matchup_id']
            if matchup_id not in seen_matchups:
                team1 = row['team_name']
                opponent = row['opponent_name'] or 'BYE'
                score1 = row['points']
                score2 = row['opponent_points'] or 0

                matchup_label = f"{team1} ({score1:.1f}) vs {opponent} ({score2:.1f})"
                matchup_options.append((matchup_label, matchup_id))
                seen_matchups.add(matchup_id)

        if not matchup_options:
            st.error("No complete matchups found.")
            return selected_week, None

        with col2:
            selected_matchup_label = st.selectbox(
                "Select Matchup",
                options=[label for label, _ in matchup_options]
            )

        selected_matchup_id = next(mid for label, mid in matchup_options if label == selected_matchup_label)

        return selected_week, selected_matchup_id

    def render_matchup_header(self, team1_data: pd.Series, team2_data: pd.Series, week: int):
        """Render matchup header with better styling."""
        team1_name = team1_data['team_name']
        team2_name = team2_data['team_name']
        team1_score = team1_data['points']
        team2_score = team2_data['points']

        st.markdown(f"### Week {week} Matchup")

        # Create visual matchup header
        col1, col2, col3 = st.columns([3, 1, 3])

        with col1:
            if team1_score > team2_score:
                st.success(f"🏆 **{team1_name}**")
                st.metric("Final Score", f"{team1_score:.1f}")
            elif team1_score < team2_score:
                st.error(f"**{team1_name}**")
                st.metric("Final Score", f"{team1_score:.1f}")
            else:
                st.info(f"**{team1_name}**")
                st.metric("Final Score", f"{team1_score:.1f}")

        with col2:
            st.markdown("<div style='text-align: center; padding-top: 20px;'><h2>VS</h2></div>",
                        unsafe_allow_html=True)

        with col3:
            if team2_score > team1_score:
                st.success(f"🏆 **{team2_name}**")
                st.metric("Final Score", f"{team2_score:.1f}")
            elif team2_score < team1_score:
                st.error(f"**{team2_name}**")
                st.metric("Final Score", f"{team2_score:.1f}")
            else:
                st.info(f"**{team2_name}**")
                st.metric("Final Score", f"{team2_score:.1f}")

    def render_score_analysis(self, team1_analysis: Dict, team2_analysis: Dict, team1_name: str, team2_name: str):
        """Render improved side-by-side score analysis."""
        st.markdown("### Performance Breakdown")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f"#### {team1_name}")

            # Key metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Actual Score", f"{team1_analysis['actual_score']:.1f}")
                st.metric("vs League Median", f"{team1_analysis['vs_expected']:.1f}")
            with metrics_col2:
                st.metric("Optimal Score", f"{team1_analysis['optimal_score']:.1f}")
                st.metric("Efficiency", f"{team1_analysis['lineup_efficiency']:.1%}")
            with metrics_col3:
                st.metric("Expected Wins vs. League", f"{team1_analysis['expected_win_pct']:.1f}")
                st.metric("Points Left on Bench", f"{team1_analysis['points_left_on_bench']:.1f}")

        with col2:
            st.markdown(f"#### {team2_name}")

            # Key metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Actual Score", f"{team2_analysis['actual_score']:.1f}")
                st.metric("vs League Median", f"{team2_analysis['vs_expected']:.1f}")
            with metrics_col2:
                st.metric("Optimal Score", f"{team2_analysis['optimal_score']:.1f}")
                st.metric("Efficiency", f"{team2_analysis['lineup_efficiency']:.1%}")
            with metrics_col3:
                st.metric("Expected Wins vs. League", f"{team2_analysis['expected_win_pct']:.1f}")
                st.metric("Points Left on Bench", f"{team2_analysis['points_left_on_bench']:.1f}")

    def render_player_performances(self, team1_analysis: Dict, team2_analysis: Dict, team1_name: str, team2_name: str):
        """Render improved player performance comparison."""
        st.markdown("### Player Performance Analysis")
        st.markdown("*Comparing actual points vs positional average for the week*")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f"#### {team1_name}")
            team1_comparison = team1_analysis['player_comparison']
            if not team1_comparison.empty:
                # Starters section
                starters_df = team1_comparison[team1_comparison['is_starter']].head(12)
                if not starters_df.empty:
                    st.markdown("**🏃 Starters**")

                    # Color code based on performance
                    styled_starters = starters_df.copy()

                    def style_performance(row):
                        if row['difference'] > 5:
                            return ['background-color: #d4edda'] * len(row)  # Green for good performance
                        elif row['difference'] < -5:
                            return ['background-color: #f8d7da'] * len(row)  # Red for poor performance
                        else:
                            return [''] * len(row)

                    display_df = starters_df[
                        ['name', 'position', 'actual_points', 'position_avg', 'difference', 'performance_pct']]
                    display_df.columns = ['Player', 'Pos', 'Actual', 'Pos Avg', '+/-', '% of Avg']
                    st.dataframe(display_df, hide_index=True, use_container_width=True)

                # Top bench performers
                bench_df = team1_comparison[~team1_comparison['is_starter']].head(5)
                if not bench_df.empty:
                    st.markdown("**🪑 Top Bench Players**")
                    display_df = bench_df[['name', 'position', 'actual_points', 'position_avg', 'difference']]
                    display_df.columns = ['Player', 'Pos', 'Actual', 'Pos Avg', '+/-']
                    st.dataframe(display_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown(f"#### {team2_name}")
            team2_comparison = team2_analysis['player_comparison']
            if not team2_comparison.empty:
                # Starters section
                starters_df = team2_comparison[team2_comparison['is_starter']].head(12)
                if not starters_df.empty:
                    st.markdown("**🏃 Starters**")
                    display_df = starters_df[
                        ['name', 'position', 'actual_points', 'position_avg', 'difference', 'performance_pct']]
                    display_df.columns = ['Player', 'Pos', 'Actual', 'Pos Avg', '+/-', '% of Avg']
                    st.dataframe(display_df, hide_index=True, use_container_width=True)

                # Top bench performers
                bench_df = team2_comparison[~team2_comparison['is_starter']].head(5)
                if not bench_df.empty:
                    st.markdown("**🪑 Top Bench Players**")
                    display_df = bench_df[['name', 'position', 'actual_points', 'position_avg', 'difference']]
                    display_df.columns = ['Player', 'Pos', 'Actual', 'Pos Avg', '+/-']
                    st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Week's best and worst performers
        st.markdown("#### Weekly Performance Leaders")

        col1, col2 = st.columns(2)

        # Combine all players from both teams
        all_players = pd.concat([team1_analysis['player_comparison'], team2_analysis['player_comparison']])

        with col1:
            st.markdown("**🔥 Biggest Overperformers**")
            top_over = all_players.nlargest(5, 'difference')[
                ['name', 'position', 'actual_points', 'position_avg', 'difference']]
            top_over.columns = ['Player', 'Pos', 'Actual', 'Pos Avg', '+/-']
            st.dataframe(top_over, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**❄️ Biggest Underperformers**")
            top_under = all_players.nsmallest(5, 'difference')[
                ['name', 'position', 'actual_points', 'position_avg', 'difference']]
            top_under.columns = ['Player', 'Pos', 'Actual', 'Pos Avg', '+/-']
            st.dataframe(top_under, hide_index=True, use_container_width=True)

        # Performance distribution chart
        st.markdown("#### Performance Distribution")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{team1_name} Starters', f'{team2_name} Starters'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Team 1 starters performance
        team1_starters = team1_analysis['player_comparison'][team1_analysis['player_comparison']['is_starter']]
        if not team1_starters.empty:
            fig.add_trace(go.Bar(
                x=team1_starters['position'],
                y=team1_starters['difference'],
                name=f'{team1_name}',
                marker_color=['green' if x > 0 else 'red' for x in team1_starters['difference']],
                showlegend=False
            ), row=1, col=1)

        # Team 2 starters performance
        team2_starters = team2_analysis['player_comparison'][team2_analysis['player_comparison']['is_starter']]
        if not team2_starters.empty:
            fig.add_trace(go.Bar(
                x=team2_starters['position'],
                y=team2_starters['difference'],
                name=f'{team2_name}',
                marker_color=['green' if x > 0 else 'red' for x in team2_starters['difference']],
                showlegend=False
            ), row=1, col=2)

        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.update_layout(height=400, title="Starter Performance vs Positional Average")
        fig.update_yaxes(title_text="Points Above/Below Positional Average")

        st.plotly_chart(fig, use_container_width=True)

    def render(self, league_id: str, season: str):
        """Main render function."""
        # Matchup selection
        selected_week, selected_matchup_id = self.render_matchup_selector(league_id)

        if not selected_week or not selected_matchup_id:
            return

        # Load data
        with st.spinner("Loading matchup data..."):
            week_matchups = self.get_week_matchups(league_id, selected_week)
            league_scores = self.get_league_week_scores(league_id, selected_week)
            players_df = self.get_players_table()
            position_averages = self.get_week_position_averages(league_id, selected_week)

        # Filter to selected matchup
        matchup_teams = week_matchups[week_matchups['matchup_id'] == selected_matchup_id]

        if len(matchup_teams) != 2:
            st.error("Invalid matchup selection.")
            return

        team1_data = matchup_teams.iloc[0]
        team2_data = matchup_teams.iloc[1]

        # Calculate analysis
        team1_analysis = self.calculate_matchup_analysis(
            team1_data, team2_data, league_scores, players_df, position_averages)
        team2_analysis = self.calculate_matchup_analysis(
            team2_data, team1_data, league_scores, players_df, position_averages)

        # Render sections
        self.render_matchup_header(team1_data, team2_data, selected_week)
        st.divider()

        self.render_score_analysis(team1_analysis, team2_analysis, team1_data['team_name'], team2_data['team_name'])
        st.divider()

        self.render_player_performances(team1_analysis, team2_analysis, team1_data['team_name'],
                                        team2_data['team_name'])