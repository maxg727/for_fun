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

    def calculate_matchup_analysis(self, matchup_data: pd.Series, opponent_data: pd.Series,
                                   league_scores: pd.DataFrame, players_df: pd.DataFrame) -> Dict[str, Any]:
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
        else:
            analysis['expected_score'] = actual_score
            analysis['vs_expected'] = 0.0

        # Replace the existing player performance lines in calculate_matchup_analysis with:
        analysis['player_comparison'] = self._get_player_performance_comparison(players_points, starters, players_df)

        return analysis

    def _calculate_player_expected_points(self, players_points: Dict, players_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate expected points based on player's season average or position average."""
        # For now, use league median by position as expected points
        # You could enhance this with historical averages or projections
        pos_map = dict(zip(players_df['player_id'], players_df['position']))

        expected_points = {}
        for pid in players_points.keys():
            position = pos_map.get(pid, 'UNK')
            # Simple expected points based on position (you can make this more sophisticated)
            position_expected = {
                'QB': 18.0, 'RB': 12.0, 'WR': 11.0, 'TE': 8.0,
                'K': 7.0, 'DEF': 8.0, 'UNK': 5.0
            }
            expected_points[pid] = position_expected.get(position, 5.0)

        return expected_points

    def _get_player_performance_comparison(self, players_points: Dict, starters: List,
                                           players_df: pd.DataFrame) -> pd.DataFrame:
        """Get all players' actual vs expected performance."""
        if not players_points:
            return pd.DataFrame()

        name_map = dict(zip(players_df['player_id'], players_df['full_name']))
        pos_map = dict(zip(players_df['player_id'], players_df['position']))
        expected_points = self._calculate_player_expected_points(players_points, players_df)

        performance_data = []
        for pid, actual_pts in players_points.items():
            expected_pts = expected_points.get(pid, 5.0)
            diff = actual_pts - expected_pts

            performance_data.append({
                'player_id': pid,
                'name': name_map.get(pid, 'Unknown'),
                'position': pos_map.get(pid, 'UNK'),
                'actual_points': round(actual_pts, 1),
                'expected_points': round(expected_pts, 1),
                'difference': round(diff, 1),
                'is_starter': pid in starters
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
        """Render matchup header."""
        team1_name = team1_data['team_name']
        team2_name = team2_data['team_name']
        team1_score = team1_data['points']
        team2_score = team2_data['points']

        st.subheader(f"Week {week} Matchup")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            st.metric(team1_name, f"{team1_score:.1f}")

        with col2:
            if team1_score > team2_score:
                st.success("Winner")
                st.caption(f"Margin: {abs(team1_score - team2_score):.1f}")
            elif team2_score > team1_score:
                st.error("Loser")
                st.caption(f"Margin: {abs(team2_score - team1_score):.1f}")
            else:
                st.info("Tie")

        with col3:
            st.metric(team2_name, f"{team2_score:.1f}")

    def render_score_analysis(self, team1_analysis: Dict, team2_analysis: Dict, team1_name: str, team2_name: str):
        """Render score analysis comparison."""
        st.subheader("Performance Breakdown")

        # Metrics comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{team1_name}**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Actual Score", f"{team1_analysis['actual_score']:.1f}")
                st.metric("Optimal Score", f"{team1_analysis['optimal_score']:.1f}")
            with col_b:
                st.metric("Efficiency", f"{team1_analysis['lineup_efficiency']:.1%}")
                st.metric("Bench Points Left", f"{team1_analysis['points_left_on_bench']:.1f}")

        with col2:
            st.markdown(f"**{team2_name}**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Actual Score", f"{team2_analysis['actual_score']:.1f}")
                st.metric("Optimal Score", f"{team2_analysis['optimal_score']:.1f}")
            with col_b:
                st.metric("Efficiency", f"{team2_analysis['lineup_efficiency']:.1%}")
                st.metric("Bench Points Left", f"{team2_analysis['points_left_on_bench']:.1f}")

        # Visual comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Score Comparison', 'Efficiency Comparison')
        )

        # Score comparison
        categories = ['Actual', 'Expected', 'Optimal']
        team1_values = [team1_analysis['actual_score'], team1_analysis['expected_score'],
                        team1_analysis['optimal_score']]
        team2_values = [team2_analysis['actual_score'], team2_analysis['expected_score'],
                        team2_analysis['optimal_score']]

        fig.add_trace(go.Bar(
            name=team1_name,
            x=categories,
            y=team1_values,
            marker_color='lightblue'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name=team2_name,
            x=categories,
            y=team2_values,
            marker_color='lightcoral'
        ), row=1, col=1)

        # Efficiency comparison
        eff_categories = ['Efficiency %', 'Bench Points']
        team1_eff = [team1_analysis['lineup_efficiency'] * 100, team1_analysis['points_left_on_bench']]
        team2_eff = [team2_analysis['lineup_efficiency'] * 100, team2_analysis['points_left_on_bench']]

        fig.add_trace(go.Bar(
            name=team1_name,
            x=eff_categories,
            y=team1_eff,
            marker_color='lightblue',
            showlegend=False
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            name=team2_name,
            x=eff_categories,
            y=team2_eff,
            marker_color='lightcoral',
            showlegend=False
        ), row=1, col=2)

        fig.update_layout(height=400, barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    def render_player_performances(self, team1_analysis: Dict, team2_analysis: Dict, team1_name: str, team2_name: str):
        """Render player actual vs expected performance comparison."""
        st.subheader("Player Performance: Actual vs Expected")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{team1_name}**")
            team1_comparison = team1_analysis['player_comparison']
            if not team1_comparison.empty:
                # Show starters first, then bench
                starters_df = team1_comparison[team1_comparison['is_starter']].head(10)
                bench_df = team1_comparison[~team1_comparison['is_starter']].head(5)

                st.markdown("*Starters*")
                display_df = starters_df[['name', 'position', 'actual_points', 'expected_points', 'difference']]
                st.dataframe(display_df, hide_index=True, use_container_width=True)

                if not bench_df.empty:
                    st.markdown("*Top Bench*")
                    display_df = bench_df[['name', 'position', 'actual_points', 'expected_points', 'difference']]
                    st.dataframe(display_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown(f"**{team2_name}**")
            team2_comparison = team2_analysis['player_comparison']
            if not team2_comparison.empty:
                # Show starters first, then bench
                starters_df = team2_comparison[team2_comparison['is_starter']].head(10)
                bench_df = team2_comparison[~team2_comparison['is_starter']].head(5)

                st.markdown("*Starters*")
                display_df = starters_df[['name', 'position', 'actual_points', 'expected_points', 'difference']]
                st.dataframe(display_df, hide_index=True, use_container_width=True)

                if not bench_df.empty:
                    st.markdown("*Top Bench*")
                    display_df = bench_df[['name', 'position', 'actual_points', 'expected_points', 'difference']]
                    st.dataframe(display_df, hide_index=True, use_container_width=True)

        # Summary of biggest over/under performers
        st.markdown("#### Biggest Over/Under Performers")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Biggest Overperformers**")
            all_players = pd.concat([team1_analysis['player_comparison'], team2_analysis['player_comparison']])
            top_over = all_players.nlargest(5, 'difference')[
                ['name', 'position', 'actual_points', 'expected_points', 'difference']]
            st.dataframe(top_over, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**Biggest Underperformers**")
            top_under = all_players.nsmallest(5, 'difference')[
                ['name', 'position', 'actual_points', 'expected_points', 'difference']]
            st.dataframe(top_under, hide_index=True, use_container_width=True)

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

        # Filter to selected matchup
        matchup_teams = week_matchups[week_matchups['matchup_id'] == selected_matchup_id]

        if len(matchup_teams) != 2:
            st.error("Invalid matchup selection.")
            return

        team1_data = matchup_teams.iloc[0]
        team2_data = matchup_teams.iloc[1]

        # Calculate analysis
        team1_analysis = self.calculate_matchup_analysis(team1_data, team2_data, league_scores, players_df)
        team2_analysis = self.calculate_matchup_analysis(team2_data, team1_data, league_scores, players_df)

        # Render sections
        self.render_matchup_header(team1_data, team2_data, selected_week)
        st.divider()

        self.render_score_analysis(team1_analysis, team2_analysis, team1_data['team_name'], team2_data['team_name'])
        st.divider()

        self.render_player_performances(team1_analysis, team2_analysis, team1_data['team_name'],
                                        team2_data['team_name'])