import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
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


class TeamAnalysisModule:
    """
    Optimized Streamlit Team Analysis module for Sleeper leagues.

    Key features:
      - Expected vs Actual performance (all-play expected wins, luck index)
      - Lineup efficiency (actual vs optimal starters) and perfect lineup rate
      - Positional record vs opponents (per position group)
      - Points Above Replacement (PAR) for each rostered player
      - Streamlined UI with better performance and cleaner presentation
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection = None

    def __enter__(self):
        self._connection = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connection:
            self._connection.close()

    @property
    def connection(self):
        """Get database connection, creating one if needed."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
        return self._connection

    # ---------------------------
    # Data access helpers
    # ---------------------------

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_team_list(_self, league_id: str) -> List[Dict[str, Any]]:
        """Return list of active teams in a league (for the selector)."""
        query = """
            SELECT 
                r.roster_id,
                r.owner_id,
                COALESCE(u.display_name, 'Team ' || r.roster_id) as team_name,
                r.settings
            FROM rosters r
            LEFT JOIN users u ON r.owner_id = u.user_id
            WHERE r.league_id = ? AND r.owner_id IS NOT NULL
            ORDER BY team_name
        """
        df = pd.read_sql(query, _self.connection, params=(league_id,))

        teams = []
        for _, row in df.iterrows():
            settings = _safe_json_loads(row.get('settings'), "dict")
            teams.append({
                'roster_id': int(row['roster_id']),
                'owner_id': row['owner_id'],
                'team_name': row['team_name'],
                'wins': int(settings.get('wins', 0) or 0),
                'losses': int(settings.get('losses', 0) or 0),
                'ties': int(settings.get('ties', 0) or 0),
                'points_for': float(settings.get('fpts', 0) or 0),
            })
        return teams

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_players_table(_self) -> pd.DataFrame:
        """Load player metadata with optimized query."""
        query = """
            SELECT DISTINCT player_id, 
                   COALESCE(full_name, 'Unknown') as full_name, 
                   COALESCE(position, 'UNK') as position 
            FROM players 
            WHERE player_id IS NOT NULL
        """
        return pd.read_sql(query, _self.connection)

    @st.cache_data(ttl=300)
    def get_team_season_data(_self, league_id: str, roster_id: int) -> Dict[str, pd.DataFrame]:
        """
        Return roster row and matchups with opponent stats for the team.
        Optimized with a single query for matchups and opponents.
        """
        # Get roster info
        roster_query = """
            SELECT r.*, COALESCE(u.display_name, 'Team ' || r.roster_id) as team_name
            FROM rosters r
            LEFT JOIN users u ON r.owner_id = u.user_id
            WHERE r.league_id = ? AND r.roster_id = ?
        """
        roster_df = pd.read_sql(roster_query, _self.connection, params=(league_id, roster_id))

        # Optimized matchups query with CTE for better performance
        matchups_query = """
            WITH team_matchups AS (
                SELECT league_id, week, matchup_id, roster_id, points, 
                       starters, starters_points, players_points
                FROM matchups 
                WHERE league_id = ? AND roster_id = ?
            ),
            opponent_matchups AS (
                SELECT m.league_id, m.week, m.matchup_id, m.roster_id, m.points,
                       m.starters, m.players_points,
                       COALESCE(u.display_name, 'Team ' || m.roster_id) as opponent_name
                FROM matchups m
                LEFT JOIN rosters r ON m.roster_id = r.roster_id AND m.league_id = r.league_id
                LEFT JOIN users u ON r.owner_id = u.user_id
                WHERE m.league_id = ?
            )
            SELECT 
                tm.*,
                om.roster_id as opponent_roster_id,
                om.points as opponent_points,
                om.starters as opponent_starters_json,
                om.players_points as opponent_players_points_json,
                om.opponent_name
            FROM team_matchups tm
            LEFT JOIN opponent_matchups om 
                ON tm.league_id = om.league_id 
                AND tm.week = om.week 
                AND tm.matchup_id = om.matchup_id 
                AND tm.roster_id != om.roster_id
            ORDER BY tm.week
        """
        matchups_df = pd.read_sql(matchups_query, _self.connection, params=(league_id, roster_id, league_id))

        # Parse JSON columns efficiently
        json_columns = {
            'starters': 'list',
            'starters_points': 'list',
            'players_points': 'dict',
            'opponent_starters_json': 'list',
            'opponent_players_points_json': 'dict',
        }

        for col, default_type in json_columns.items():
            if col in matchups_df.columns:
                matchups_df[col] = matchups_df[col].apply(
                    lambda x: _safe_json_loads(x, default_type)
                )

        # Get recent transactions (simplified query)
        transactions_query = """
            SELECT type, created, adds, drops 
            FROM transactions
            WHERE league_id = ? 
              AND (roster_ids LIKE '%' || ? || '%' OR creator = (
                  SELECT owner_id FROM rosters 
                  WHERE league_id = ? AND roster_id = ?
              ))
            ORDER BY created DESC 
            LIMIT 20
        """
        transactions_df = pd.read_sql(
            transactions_query, _self.connection,
            params=(league_id, str(roster_id), league_id, roster_id)
        )

        return {
            'roster': roster_df,
            'matchups': matchups_df,
            'transactions': transactions_df
        }

    @st.cache_data(ttl=300)
    def get_league_weekly_scores(_self, league_id: str) -> pd.DataFrame:
        """All team-week scores for expected wins calculation."""
        return pd.read_sql(
            "SELECT week, roster_id, points FROM matchups WHERE league_id = ? AND points IS NOT NULL",
            _self.connection, params=(league_id,)
        )

    # ---------------------------
    # Helper method to filter played games
    # ---------------------------

    def _filter_played_games(self, matchups_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out unplayed games (future weeks).
        A game is considered played if:
        1. Both teams have non-null points
        2. At least one team scored > 0 points (handles edge case of actual 0-0 ties)
        3. Has valid opponent data
        """
        if matchups_df.empty:
            return matchups_df

        # Filter for games with valid point data
        valid = matchups_df[
            matchups_df['points'].notna() &
            matchups_df['opponent_points'].notna()
            ].copy()

        # Additional filter: exclude games where both teams scored exactly 0
        # (likely unplayed games, as real 0-0 ties are extremely rare in fantasy)
        played_games = valid[
            (valid['points'] > 0) | (valid['opponent_points'] > 0)
            ].copy()

        return played_games

    # ---------------------------
    # Optimized Metrics
    # ---------------------------

    def calculate_performance_metrics(self, matchups_df: pd.DataFrame, league_scores: pd.DataFrame) -> Dict[str, Any]:
        """Optimized performance metrics calculation - only includes played games."""
        if matchups_df.empty:
            return self._empty_metrics()

        # Filter to only played games
        valid = self._filter_played_games(matchups_df)
        if valid.empty:
            return self._empty_metrics()

        # Vectorized W/L/T calculation
        opponent_points = valid['opponent_points'].fillna(0)
        points = valid['points']

        wins = (points > opponent_points).sum()
        losses = (points < opponent_points).sum()
        ties = (points == opponent_points).sum()
        games = len(valid)

        # Efficient expected wins calculation - only use played games
        exp_wins_total = 0.0
        for week in valid['week'].unique():
            week_data = valid[valid['week'] == week]
            if len(week_data) == 0:
                continue

            # Filter league scores to only include games with actual scores > 0
            week_league_scores = league_scores[league_scores['week'] == week]
            week_league_scores = week_league_scores[week_league_scores['points'] > 0]
            week_scores = week_league_scores['points']

            if week_scores.empty:
                continue

            for points_scored in week_data['points']:
                if points_scored > 0:  # Only calculate for actual played games
                    lower = (week_scores < points_scored).sum()
                    equal = (week_scores == points_scored).sum()
                    total_teams = len(week_scores)

                    if total_teams > 1:
                        exp_wins_total += (lower + 0.5 * max(0, equal - 1)) / (total_teams - 1)

        # Calculate metrics
        win_pct = (wins + 0.5 * ties) / max(1, games)
        luck_index = (wins + 0.5 * ties) - exp_wins_total

        return {
            'games': games,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_pct': win_pct,
            'total_points': float(points.sum()),
            'avg_points': float(points.mean()),
            'std_points': float(points.std()),
            'expected_wins': exp_wins_total,
            'luck_index': luck_index,
            'highest_score': float(points.max()),
            'lowest_score': float(points.min()),
            'median_score': float(points.median()),
            'current_streak': self._calculate_streak(valid)
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'games': 0, 'wins': 0, 'losses': 0, 'ties': 0, 'win_pct': 0.0,
            'total_points': 0.0, 'avg_points': 0.0, 'std_points': 0.0,
            'expected_wins': 0.0, 'luck_index': 0.0, 'highest_score': 0.0,
            'lowest_score': 0.0, 'median_score': 0.0, 'current_streak': "0"
        }

    def _calculate_streak(self, valid_df: pd.DataFrame) -> str:
        """Calculate current win/loss streak - only for played games."""
        if valid_df.empty or 'opponent_points' not in valid_df.columns:
            return "0"

        results = []
        for _, row in valid_df.sort_values('week').iterrows():
            if pd.isna(row['opponent_points']) or pd.isna(row['points']):
                continue
            # Skip unplayed games (both teams scored 0)
            if row['points'] == 0 and row['opponent_points'] == 0:
                continue

            if row['points'] > row['opponent_points']:
                results.append('W')
            elif row['points'] < row['opponent_points']:
                results.append('L')
            else:
                results.append('T')

        if not results:
            return "0"

        current = results[-1]
        count = 1
        for i in range(len(results) - 2, -1, -1):
            if results[i] == current:
                count += 1
            else:
                break
        return f"{current}{count}"

    def calculate_lineup_efficiency(self, matchups_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimized lineup efficiency calculation - only for played games."""
        if matchups_df.empty:
            return {
                "avg_efficiency": 1.0,
                "points_left_on_bench": 0.0,
                "perfect_weeks": 0,
                "weekly": pd.DataFrame(columns=["week", "starter_points", "optimal_points", "lineup_efficiency"])
            }

        # Filter to only played games
        valid_matchups = self._filter_played_games(matchups_df)
        if valid_matchups.empty:
            return {
                "avg_efficiency": 1.0,
                "points_left_on_bench": 0.0,
                "perfect_weeks": 0,
                "weekly": pd.DataFrame(columns=["week", "starter_points", "optimal_points", "lineup_efficiency"])
            }

        weekly_data = []
        total_actual = total_optimal = total_bench_lost = perfect_weeks = 0

        for _, row in valid_matchups.iterrows():
            starters = row.get('starters', []) or []
            players_points = row.get('players_points', {}) or {}

            if not starters or not players_points:
                continue

            # Calculate actual and optimal points
            actual = sum(players_points.get(pid, 0.0) for pid in starters)
            all_scores = sorted(players_points.values(), reverse=True)
            optimal = sum(all_scores[:len(starters)]) if starters else 0.0

            efficiency = actual / optimal if optimal > 0 else 1.0
            bench_lost = max(0.0, optimal - actual)

            weekly_data.append({
                "week": int(row['week']),
                "starter_points": actual,
                "optimal_points": optimal,
                "lineup_efficiency": efficiency,
                "points_left_on_bench": bench_lost
            })

            total_actual += actual
            total_optimal += optimal
            total_bench_lost += bench_lost
            if efficiency >= 0.999:
                perfect_weeks += 1

        weekly_df = pd.DataFrame(weekly_data).sort_values('week') if weekly_data else pd.DataFrame()
        avg_efficiency = total_actual / total_optimal if total_optimal > 0 else 1.0

        return {
            "avg_efficiency": avg_efficiency,
            "points_left_on_bench": total_bench_lost,
            "perfect_weeks": perfect_weeks,
            "weekly": weekly_df
        }

    def calculate_keeper_analysis(self, matchups_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate keeper value by comparing positional rank vs draft position."""
        if matchups_df.empty:
            return pd.DataFrame(
                columns=["player_id", "full_name", "position", "total_pts", "pos_rank", "draft_rank", "keeper_value",
                         "keeper_grade"])

        # Get draft data - you'll need to add this query to get draft positions
        with self.connection as conn:
            draft_query = """
                SELECT player_id, pick_no, round 
                FROM draft_picks 
                WHERE league_id = ? AND roster_id = ?
            """
            # Note: This assumes you have draft data - if not, you'll need to add this table/data

        # Filter to only played games
        valid_matchups = self._filter_played_games(matchups_df)

        # Calculate total points per player from matchups
        player_totals = {}
        for _, row in valid_matchups.iterrows():
            players_points = row.get('players_points', {}) or {}
            for pid, pts in players_points.items():
                player_totals[pid] = player_totals.get(pid, 0) + float(pts)

        # Get position rankings - need all league data for this
        pos_rankings = self._calculate_positional_rankings(player_totals, players_df)

        # Combine with draft data and calculate keeper value
        keeper_data = []
        for pid, total_pts in player_totals.items():
            player_info = players_df[players_df['player_id'] == pid]
            if player_info.empty:
                continue

            pos = player_info.iloc[0]['position']
            name = player_info.iloc[0]['full_name']

            pos_rank = pos_rankings.get(pid, 999)
            # You'll need draft_rank from your draft data
            draft_rank = self._get_draft_position(pid)  # Implement this

            if draft_rank and pos_rank < 999:
                keeper_value = draft_rank - pos_rank  # Higher is better
                grade = self._get_keeper_grade(keeper_value)

                keeper_data.append({
                    "player_id": pid,
                    "full_name": name,
                    "position": pos,
                    "total_pts": total_pts,
                    "pos_rank": pos_rank,
                    "draft_rank": draft_rank,
                    "keeper_value": keeper_value,
                    "keeper_grade": grade
                })

        return pd.DataFrame(keeper_data).sort_values("keeper_value", ascending=False)

    def _get_keeper_grade(self, keeper_value: int) -> str:
        """Assign letter grade based on keeper value."""
        if keeper_value >= 24:
            return "A+"
        elif keeper_value >= 12:
            return "A"
        elif keeper_value >= 6:
            return "B"
        elif keeper_value >= 0:
            return "C"
        else:
            return "D"

    def calculate_positional_record(self, matchups_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized positional record calculation - only for played games."""
        if matchups_df.empty:
            return pd.DataFrame(columns=["Position", "Games", "Record", "PF", "PA", "Diff", "Avg PF", "Avg PA"])

        # Filter to only played games
        valid_matchups = self._filter_played_games(matchups_df)
        if valid_matchups.empty:
            return pd.DataFrame(columns=["Position", "Games", "Record", "PF", "PA", "Diff", "Avg PF", "Avg PA"])

        pos_map = dict(zip(players_df['player_id'], players_df['position']))
        position_stats = {}

        for _, row in valid_matchups.iterrows():
            my_starters = row.get('starters', []) or []
            opp_starters = row.get('opponent_starters_json', []) or []
            my_points = row.get('players_points', {}) or {}
            opp_points = row.get('opponent_players_points_json', {}) or {}

            # Aggregate by position
            my_pos_totals = {}
            opp_pos_totals = {}

            for pid in my_starters:
                pos = pos_map.get(pid, 'UNK')
                my_pos_totals[pos] = my_pos_totals.get(pos, 0.0) + my_points.get(pid, 0.0)

            for pid in opp_starters:
                pos = pos_map.get(pid, 'UNK')
                opp_pos_totals[pos] = opp_pos_totals.get(pos, 0.0) + opp_points.get(pid, 0.0)

            # Update position statistics
            all_positions = set(my_pos_totals.keys()) | set(opp_pos_totals.keys())
            for pos in all_positions:
                my_pts = my_pos_totals.get(pos, 0.0)
                opp_pts = opp_pos_totals.get(pos, 0.0)

                if pos not in position_stats:
                    position_stats[pos] = {"PF": 0.0, "PA": 0.0, "games": 0, "wins": 0, "losses": 0, "ties": 0}

                stats = position_stats[pos]
                stats["PF"] += my_pts
                stats["PA"] += opp_pts
                stats["games"] += 1

                if my_pts > opp_pts:
                    stats["wins"] += 1
                elif my_pts < opp_pts:
                    stats["losses"] += 1
                else:
                    stats["ties"] += 1

        # Convert to DataFrame
        rows = []
        for pos, stats in position_stats.items():
            games = stats["games"]
            if games == 0:
                continue

            record = f"{stats['wins']}-{stats['losses']}"
            if stats['ties'] > 0:
                record += f"-{stats['ties']}"

            rows.append({
                "Position": pos,
                "Games": games,
                "Record": record,
                "PF": round(stats["PF"], 2),
                "PA": round(stats["PA"], 2),
                "Diff": round(stats["PF"] - stats["PA"], 2),
                "Avg PF": round(stats["PF"] / games, 2),
                "Avg PA": round(stats["PA"] / games, 2)
            })

        return pd.DataFrame(rows).sort_values(["Diff", "PF"], ascending=[False, False])

    def calculate_par(self, matchups_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized Points Above Replacement calculation - only for played games."""
        if matchups_df.empty:
            return pd.DataFrame(
                columns=["player_id", "position", "weeks", "total_pts", "PAR_total", "PAR_ppw", "full_name"])

        # Filter to only played games
        valid_matchups = self._filter_played_games(matchups_df)
        if valid_matchups.empty:
            return pd.DataFrame(
                columns=["player_id", "position", "weeks", "total_pts", "PAR_total", "PAR_ppw", "full_name"])

        pos_map = dict(zip(players_df['player_id'], players_df['position']))
        name_map = dict(zip(players_df['player_id'], players_df['full_name']))

        # Collect player statistics and position point distributions
        player_stats = {}
        position_points = {}

        for _, row in valid_matchups.iterrows():
            players_points = row.get('players_points', {}) or {}

            for pid, points in players_points.items():
                pos = pos_map.get(pid, 'UNK')
                points = float(points)

                # Update player stats
                if pid not in player_stats:
                    player_stats[pid] = {"position": pos, "total_points": 0.0, "weeks": 0}

                player_stats[pid]["total_points"] += points
                player_stats[pid]["weeks"] += 1

                # Collect points for replacement calculation
                if pos not in position_points:
                    position_points[pos] = []
                position_points[pos].append(points)

        # Calculate replacement levels (25th percentile per position)
        replacement_levels = {}
        for pos, points_list in position_points.items():
            if points_list:
                replacement_levels[pos] = float(np.percentile(points_list, 25))
            else:
                replacement_levels[pos] = 0.0

        # Build result DataFrame
        rows = []
        for pid, stats in player_stats.items():
            pos = stats["position"]
            weeks = stats["weeks"]
            total_points = stats["total_points"]
            replacement = replacement_levels.get(pos, 0.0)
            par_total = total_points - (weeks * replacement)

            rows.append({
                "player_id": pid,
                "position": pos,
                "weeks": weeks,
                "total_pts": total_points,
                "PAR_total": par_total,
                "PAR_ppw": par_total / weeks if weeks > 0 else 0.0,
                "full_name": name_map.get(pid, "Unknown")
            })

        return pd.DataFrame(rows).sort_values("PAR_total", ascending=False)

    # ---------------------------
    # Optimized Visuals
    # ---------------------------

    def render_team_header(self, team_data: Dict[str, pd.DataFrame], metrics: Dict[str, Any]):
        """Streamlined team header with key metrics."""
        if team_data['roster'].empty:
            st.error("No roster data found.")
            return

        roster_info = team_data['roster'].iloc[0]
        team_name = roster_info.get('team_name', 'Unknown Team')

        st.title(f"🏈 {team_name}")

        # Key metrics in a clean layout
        col1, col2, col3, col4, col5 = st.columns(5)

        wins = metrics.get('wins', 0)
        losses = metrics.get('losses', 0)
        ties = metrics.get('ties', 0)
        streak = metrics.get('current_streak', '')

        with col1:
            record = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
            st.metric("Record", record, streak if streak != "0" else None)

        with col2:
            st.metric("Win %", f"{metrics.get('win_pct', 0):.1%}")

        with col3:
            st.metric("Avg Points", f"{metrics.get('avg_points', 0):.1f}")

        with col4:
            st.metric("Expected Wins", f"{metrics.get('expected_wins', 0):.1f}")

        with col5:
            luck = metrics.get('luck_index', 0)
            luck_color = "normal" if abs(luck) < 1 else ("inverse" if luck < 0 else "normal")
            st.metric("Luck Index", f"{luck:+.1f}", delta_color=luck_color)

    def render_weekly_performance(self, matchups_df: pd.DataFrame):
        """Optimized weekly performance chart - only shows played games."""
        if matchups_df.empty:
            st.info("No matchup data available.")
            return

        st.subheader("📈 Weekly Performance")

        # Filter to only played games
        df = self._filter_played_games(matchups_df)
        if df.empty:
            st.info("No played games yet.")
            return

        # Create win/loss indicators
        df['result'] = 'Unknown'
        has_opponent = df['opponent_points'].notna()
        df.loc[has_opponent & (df['points'] > df['opponent_points']), 'result'] = 'Win'
        df.loc[has_opponent & (df['points'] < df['opponent_points']), 'result'] = 'Loss'
        df.loc[has_opponent & (df['points'] == df['opponent_points']), 'result'] = 'Tie'

        # Create plot
        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # Your points line
        fig.add_trace(go.Scatter(
            x=df['week'],
            y=df['points'],
            name='Your Points',
            mode='lines+markers',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8),
            hovertemplate='Week %{x}<br>Points: %{y:.1f}<extra></extra>'
        ))

        # Opponent points (if available)
        if 'opponent_points' in df.columns and df['opponent_points'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['week'],
                y=df['opponent_points'],
                name='Opponent Points',
                mode='lines+markers',
                line=dict(width=2, dash='dash', color='#ff7f0e'),
                marker=dict(size=6),
                hovertemplate='Week %{x}<br>Opponent: %{y:.1f}<extra></extra>'
            ))

        # Season average line
        avg_points = df['points'].mean()
        fig.add_hline(
            y=avg_points,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Season Avg: {avg_points:.1f}",
            annotation_position="top right"
        )

        # Win/Loss background colors
        for _, row in df.iterrows():
            if row['result'] == 'Win':
                color = 'rgba(0, 200, 0, 0.1)'
            elif row['result'] == 'Loss':
                color = 'rgba(200, 0, 0, 0.1)'
            elif row['result'] == 'Tie':
                color = 'rgba(200, 200, 0, 0.1)'
            else:
                continue

            fig.add_vrect(
                x0=row['week'] - 0.4,
                x1=row['week'] + 0.4,
                fillcolor=color,
                layer="below",
                line_width=0
            )

        fig.update_layout(
            title="Weekly Performance",
            xaxis_title="Week",
            yaxis_title="Points",
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            showlegend=True
        )
        fig.update_xaxes(dtick=1)

        st.plotly_chart(fig, use_container_width=True)

        # Performance summary
        if not df.empty:
            col1, col2, col3 = st.columns(3)

            with col1:
                best_week = df.loc[df['points'].idxmax()]
                st.success(f"**Best Week:** {int(best_week['week'])} — {best_week['points']:.1f} pts")

            with col2:
                worst_week = df.loc[df['points'].idxmin()]
                st.error(f"**Worst Week:** {int(worst_week['week'])} — {worst_week['points']:.1f} pts")

            with col3:
                recent = df.tail(3)['result'].value_counts()
                wins = recent.get('Win', 0)
                losses = recent.get('Loss', 0)
                ties = recent.get('Tie', 0)
                st.info(f"**Last 3:** {wins}W-{losses}L{f'-{ties}T' if ties else ''}")

    def render_efficiency_analysis(self, lineup_data: Dict[str, Any]):
        """Streamlined efficiency analysis."""
        st.subheader("🎯 Lineup Efficiency")

        col1, col2, col3 = st.columns(3)

        with col1:
            efficiency = lineup_data.get('avg_efficiency', 1.0)
            st.metric("Season Efficiency", f"{efficiency:.1%}")

        with col2:
            bench_points = lineup_data.get('points_left_on_bench', 0.0)
            st.metric("Points Left on Bench", f"{bench_points:.1f}")

        with col3:
            perfect_weeks = lineup_data.get('perfect_weeks', 0)
            weekly_df = lineup_data.get('weekly', pd.DataFrame())
            total_weeks = len(weekly_df)
            perfect_rate = perfect_weeks / total_weeks if total_weeks > 0 else 0.0
            st.metric("Perfect Weeks", f"{perfect_weeks}", f"{perfect_rate:.0%}")

        # Weekly efficiency trend
        if not weekly_df.empty and len(weekly_df) > 1:
            fig = px.line(
                weekly_df,
                x='week',
                y='lineup_efficiency',
                markers=True,
                title="Weekly Lineup Efficiency",
                labels={'lineup_efficiency': 'Efficiency', 'week': 'Week'}
            )
            fig.update_layout(height=300, yaxis_tickformat=".0%")
            fig.add_hline(y=efficiency, line_dash="dash", line_color="red",
                          annotation_text=f"Season Avg: {efficiency:.0%}")
            st.plotly_chart(fig, use_container_width=True)

    def render_positional_record(self, pos_df: pd.DataFrame):
        """Clean positional record display."""
        st.subheader("📊 Positional Performance")

        if pos_df.empty:
            st.info("No positional data available.")
            return

        # Style the dataframe for better readability
        styled_df = pos_df.style.background_gradient(
            subset=['Diff'], cmap='RdYlGn', vmin=-50, vmax=50
        ).format({
            'PF': '{:.1f}',
            'PA': '{:.1f}',
            'Diff': '{:.1f}',
            'Avg PF': '{:.1f}',
            'Avg PA': '{:.1f}'
        })

        st.dataframe(styled_df, use_container_width=True)

    def render_par_analysis(self, par_df: pd.DataFrame):
        """Streamlined PAR analysis."""
        st.subheader("📈 Points Above Replacement (PAR)")

        if par_df.empty:
            st.info("No PAR data available.")
            return

        # Show top and bottom performers
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Performers**")
            top_par = par_df.head(10)[['full_name', 'position', 'PAR_total', 'PAR_ppw']].round(2)
            st.dataframe(top_par, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**Biggest Disappointments**")
            bottom_par = par_df.tail(5)[['full_name', 'position', 'PAR_total', 'PAR_ppw']].round(2)
            st.dataframe(bottom_par, hide_index=True, use_container_width=True)

    def render_opponent_analysis(self, matchups_df: pd.DataFrame):
        """Simplified opponent analysis - only for played games."""
        if matchups_df.empty or 'opponent_name' not in matchups_df.columns:
            return

        st.subheader("🎯 Head-to-Head Records")

        # Filter to only played games
        valid = self._filter_played_games(matchups_df)
        valid = valid[valid['opponent_name'].notna()]

        if valid.empty:
            st.info("No opponent data available for played games.")
            return

        # Calculate head-to-head records
        opponent_records = []
        for opponent in valid['opponent_name'].unique():
            opp_games = valid[valid['opponent_name'] == opponent]

            wins = (opp_games['points'] > opp_games['opponent_points']).sum()
            losses = (opp_games['points'] < opp_games['opponent_points']).sum()
            ties = (opp_games['points'] == opp_games['opponent_points']).sum()

            opponent_records.append({
                'Opponent': opponent,
                'Games': len(opp_games),
                'Record': f"{wins}-{losses}{f'-{ties}' if ties > 0 else ''}",
                'Points For': opp_games['points'].sum(),
                'Points Against': opp_games['opponent_points'].sum(),
                'Point Diff': opp_games['points'].sum() - opp_games['opponent_points'].sum(),
                'Avg Scored': opp_games['points'].mean(),
                'Avg Allowed': opp_games['opponent_points'].mean()
            })

        if opponent_records:
            df = pd.DataFrame(opponent_records).sort_values('Point Diff', ascending=False)

            # Style the dataframe
            styled_df = df.style.background_gradient(
                subset=['Point Diff'], cmap='RdYlGn'
            ).format({
                'Points For': '{:.1f}',
                'Points Against': '{:.1f}',
                'Point Diff': '{:.1f}',
                'Avg Scored': '{:.1f}',
                'Avg Allowed': '{:.1f}'
            })

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

    def render_transaction_summary(self, transactions_df: pd.DataFrame):
        """Simplified transaction summary."""
        st.subheader("💼 Transaction Activity")

        if transactions_df.empty:
            st.info("No transaction data available.")
            return

        # Summary metrics
        type_counts = transactions_df['type'].value_counts()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Trades", int(type_counts.get('trade', 0)))
        with col2:
            st.metric("Waiver Claims", int(type_counts.get('waiver', 0)))
        with col3:
            st.metric("Free Agent Adds", int(type_counts.get('free_agent', 0)))

        # Activity timeline (if enough data)
        if len(transactions_df) > 5 and 'created' in transactions_df.columns:
            valid_dates = transactions_df[transactions_df['created'].notna()].copy()
            if not valid_dates.empty:
                valid_dates['date'] = pd.to_datetime(valid_dates['created'], unit='ms', errors='coerce')
                valid_dates = valid_dates.dropna(subset=['date'])

                if not valid_dates.empty:
                    valid_dates['week'] = valid_dates['date'].dt.isocalendar().week
                    weekly_counts = valid_dates.groupby('week').size().reset_index(name='transactions')

                    if len(weekly_counts) > 1:
                        fig = px.bar(
                            weekly_counts,
                            x='week',
                            y='transactions',
                            title='Weekly Transaction Activity',
                            labels={'transactions': 'Number of Transactions', 'week': 'Week'}
                        )
                        fig.update_layout(height=250, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # Main render method
    # ---------------------------

    def render(self, league_id: str, season: str):
        """Main Streamlit rendering method."""
        st.set_page_config(page_title="Team Analysis", layout="wide", initial_sidebar_state="collapsed")

        # Team selection
        teams = self.get_team_list(league_id)
        if not teams:
            st.error("No teams found for this league.")
            return

        team_names = [t['team_name'] for t in teams]
        selected_team = st.selectbox("Select Team", options=team_names, index=0)

        team_obj = next(t for t in teams if t['team_name'] == selected_team)
        roster_id = int(team_obj['roster_id'])

        # Load data with progress indicator
        with st.spinner("Loading team data..."):
            team_data = self.get_team_season_data(league_id, roster_id)
            league_scores = self.get_league_weekly_scores(league_id)
            players_df = self.get_players_table()

        # Calculate all metrics
        with st.spinner("Calculating metrics..."):
            metrics = self.calculate_performance_metrics(team_data['matchups'], league_scores)
            lineup_data = self.calculate_lineup_efficiency(team_data['matchups'])
            pos_record_df = self.calculate_positional_record(team_data['matchups'], players_df)
            par_df = self.calculate_par(team_data['matchups'], players_df)

        # Render sections
        self.render_team_header(team_data, metrics)
        st.divider()

        self.render_weekly_performance(team_data['matchups'])
        st.divider()

        self.render_efficiency_analysis(lineup_data)
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            self.render_positional_record(pos_record_df)
        with col2:
            self.render_par_analysis(par_df)

        st.divider()

        self.render_opponent_analysis(team_data['matchups'])
        st.divider()

        self.render_transaction_summary(team_data['transactions'])

        # Footer
        st.markdown("---")
        st.caption("Data updates every 5 minutes • Built with Streamlit & Plotly")