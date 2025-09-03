import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChampionshipOddsModule:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_league_settings(self, league_id):
        """Get league configuration"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT total_rosters, playoff_week_start, roster_positions, scoring_settings
            FROM leagues 
            WHERE league_id = ?
        """
        result = pd.read_sql_query(query, conn, params=(league_id,))
        conn.close()

        if not result.empty:
            return {
                'total_rosters': result.iloc[0]['total_rosters'],
                'playoff_week_start': result.iloc[0]['playoff_week_start'] or 15,
                'roster_positions': json.loads(result.iloc[0]['roster_positions'] or '[]'),
                'scoring_settings': json.loads(result.iloc[0]['scoring_settings'] or '{}')
            }
        return {}

    def get_team_records(self, league_id):
        """Get current team records and stats"""
        conn = sqlite3.connect(self.db_path)

        # Get roster info with user details
        query = """
            SELECT 
                r.roster_id,
                r.owner_id,
                u.display_name,
                u.username,
                r.settings
            FROM rosters r
            LEFT JOIN users u ON r.owner_id = u.user_id
            WHERE r.league_id = ?
        """

        rosters_df = pd.read_sql_query(query, conn, params=(league_id,))

        # Parse settings to get wins, losses, points
        records = []
        for _, row in rosters_df.iterrows():
            settings = json.loads(row['settings'] or '{}')
            records.append({
                'roster_id': row['roster_id'],
                'team_name': row['display_name'] or row['username'] or f"Team {row['roster_id']}",
                'wins': settings.get('wins', 0),
                'losses': settings.get('losses', 0),
                'ties': settings.get('ties', 0),
                'points_for': settings.get('fpts', 0) or settings.get('fpts_decimal', 0),
                'points_against': settings.get('fpts_against', 0) or settings.get('fpts_against_decimal', 0)
            })

        conn.close()
        return pd.DataFrame(records)

    def get_team_records_through_week(self, league_id, through_week=None):
        """Get team records through specified week"""
        if through_week is None:
            # Use existing get_team_records method for current season
            return self.get_team_records(league_id)

        conn = sqlite3.connect(self.db_path)

        # Get roster info with user details
        query = """
            SELECT 
                r.roster_id,
                r.owner_id,
                u.display_name,
                u.username,
                r.settings
            FROM rosters r
            LEFT JOIN users u ON r.owner_id = u.user_id
            WHERE r.league_id = ?
        """

        rosters_df = pd.read_sql_query(query, conn, params=(league_id,))

        # Calculate wins/losses through specified week using matchup data
        matchup_query = """
            SELECT 
                m1.roster_id,
                m1.week,
                m1.points as team_points,
                m2.points as opp_points
            FROM matchups m1
            JOIN matchups m2 ON m1.league_id = m2.league_id 
                AND m1.week = m2.week 
                AND m1.matchup_id = m2.matchup_id 
                AND m1.roster_id != m2.roster_id
            WHERE m1.league_id = ? AND m1.week <= ?
            ORDER BY m1.roster_id, m1.week
        """

        matchups_df = pd.read_sql_query(matchup_query, conn, params=(league_id, through_week))
        conn.close()

        # Calculate records from matchups
        records = []
        for roster_id in rosters_df['roster_id']:
            roster_info = rosters_df[rosters_df['roster_id'] == roster_id].iloc[0]
            team_matchups = matchups_df[matchups_df['roster_id'] == roster_id]

            wins = 0
            losses = 0
            ties = 0
            points_for = 0
            points_against = 0

            for _, matchup in team_matchups.iterrows():
                team_pts = matchup['team_points'] or 0
                opp_pts = matchup['opp_points'] or 0

                points_for += team_pts
                points_against += opp_pts

                if team_pts > opp_pts:
                    wins += 1
                elif team_pts < opp_pts:
                    losses += 1
                else:
                    ties += 1

            records.append({
                'roster_id': roster_id,
                'team_name': roster_info['display_name'] or roster_info['username'] or f"Team {roster_id}",
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'points_for': points_for,
                'points_against': points_against
            })

        return pd.DataFrame(records)

    def get_weekly_scores(self, league_id):
        """Get weekly scoring data for momentum analysis"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT 
                roster_id,
                week,
                points,
                points_decimal
            FROM matchups 
            WHERE league_id = ?
            ORDER BY roster_id, week
        """

        matchups_df = pd.read_sql_query(query, conn, params=(league_id,))
        conn.close()

        # Use points_decimal if available, otherwise points
        matchups_df['final_points'] = matchups_df['points_decimal'].fillna(matchups_df['points'])

        return matchups_df[['roster_id', 'week', 'final_points']]

    def get_player_performance(self, league_id):
        """Get individual player performance and rankings"""
        conn = sqlite3.connect(self.db_path)

        # Get current rosters
        roster_query = """
            SELECT roster_id, players, starters
            FROM rosters 
            WHERE league_id = ?
        """
        rosters_df = pd.read_sql_query(roster_query, conn, params=(league_id,))

        # Get player info
        players_query = """
            SELECT player_id, full_name, position, team, status
            FROM players
            WHERE status = 'Active'
        """
        players_df = pd.read_sql_query(players_query, conn)

        # Get weekly player scoring
        matchups_query = """
            SELECT 
                roster_id,
                week,
                players_points
            FROM matchups 
            WHERE league_id = ? AND players_points IS NOT NULL
        """
        matchups_df = pd.read_sql_query(matchups_query, conn, params=(league_id,))

        conn.close()

        # Calculate player season totals by team
        player_stats = {}

        for _, matchup in matchups_df.iterrows():
            roster_id = matchup['roster_id']
            week = matchup['week']

            try:
                players_points = json.loads(matchup['players_points'] or '{}')

                for player_id, points in players_points.items():
                    if player_id not in player_stats:
                        player_stats[player_id] = {'total_points': 0, 'weeks_played': 0, 'roster_id': roster_id}

                    if points and points > 0:
                        player_stats[player_id]['total_points'] += float(points)
                        player_stats[player_id]['weeks_played'] += 1
            except:
                continue

        # Convert to DataFrame and merge with player info
        if player_stats:
            player_stats_df = pd.DataFrame.from_dict(player_stats, orient='index')
            player_stats_df['player_id'] = player_stats_df.index
            player_stats_df['avg_points'] = player_stats_df['total_points'] / player_stats_df['weeks_played'].replace(0,
                                                                                                                      1)

            # Merge with player details
            player_stats_df = player_stats_df.merge(players_df, on='player_id', how='left')

            return player_stats_df

        return pd.DataFrame()

    def calculate_strength_of_schedule(self, league_id, records_df, through_week=None):
        """Calculate strength of schedule metrics"""
        conn = sqlite3.connect(self.db_path)

        # Get all matchups to determine who plays whom
        query = """
            SELECT 
                week,
                roster_id,
                matchup_id,
                points
            FROM matchups 
            WHERE league_id = ?
            ORDER BY week, matchup_id
        """

        matchups_df = pd.read_sql_query(query, conn, params=(league_id,))
        conn.close()

        if matchups_df.empty:
            return pd.DataFrame()

        # Determine current week for analysis
        if through_week:
            current_week = through_week
            analysis_matchups = matchups_df[matchups_df['week'] <= through_week]
        else:
            current_week = matchups_df['week'].max() or 1
            analysis_matchups = matchups_df

        sos_data = []

        for roster_id in records_df['roster_id']:
            team_matchups = analysis_matchups[analysis_matchups['roster_id'] == roster_id]

            # Calculate played opponents strength (historical SOS)
            played_opponent_records = []
            played_opponent_scores = []

            # Calculate remaining opponents strength (future SOS)
            remaining_opponent_records = []
            remaining_games_count = 0

            for _, matchup in matchups_df[matchups_df['roster_id'] == roster_id].iterrows():
                week = matchup['week']
                matchup_id = matchup['matchup_id']

                # Find opponent in same matchup
                opponents = matchups_df[
                    (matchups_df['week'] == week) &
                    (matchups_df['matchup_id'] == matchup_id) &
                    (matchups_df['roster_id'] != roster_id)
                    ]

                for _, opp in opponents.iterrows():
                    opp_record = records_df[records_df['roster_id'] == opp['roster_id']]
                    if not opp_record.empty:
                        opp_win_pct = opp_record.iloc[0]['wins'] / max(
                            opp_record.iloc[0]['wins'] + opp_record.iloc[0]['losses'], 1
                        )

                        if week <= current_week:
                            # Played opponents (historical SOS)
                            played_opponent_records.append(opp_win_pct)

                            # Also get opponent's scoring average for strength
                            opp_scores = analysis_matchups[
                                analysis_matchups['roster_id'] == opp['roster_id']
                                ]['points']
                            if not opp_scores.empty:
                                played_opponent_scores.append(opp_scores.mean())

                        else:
                            # Future opponents (remaining SOS)
                            remaining_opponent_records.append(opp_win_pct)
                            remaining_games_count += 1

            # Calculate metrics
            played_sos_record = np.mean(played_opponent_records) if played_opponent_records else 0.5
            played_sos_points = np.mean(played_opponent_scores) if played_opponent_scores else 0
            remaining_sos = np.mean(remaining_opponent_records) if remaining_opponent_records else 0.5

            # Overall SOS (combination of record and scoring strength)
            if played_opponent_scores and records_df['points_for'].max() > 0:
                played_sos_points_norm = played_sos_points / records_df['points_for'].mean()
                overall_played_sos = (played_sos_record * 0.6) + (played_sos_points_norm * 0.4)
            else:
                overall_played_sos = played_sos_record

            sos_data.append({
                'roster_id': roster_id,
                'played_sos_record': played_sos_record,  # Win % of played opponents
                'played_sos_points': played_sos_points,  # Avg points of played opponents
                'played_sos_overall': overall_played_sos,  # Combined played SOS
                'remaining_sos': remaining_sos,  # Win % of remaining opponents
                'remaining_games': remaining_games_count,
                'total_opponents_played': len(played_opponent_records)
            })

        return pd.DataFrame(sos_data)

    def calculate_expected_wins(self, weekly_scores_df, records_df, through_week=None):
        """Calculate expected wins based on scoring through specified week"""
        expected_wins = []

        # Filter data if through_week is specified
        if through_week:
            weekly_scores_df = weekly_scores_df[weekly_scores_df['week'] <= through_week].copy()

        for roster_id in records_df['roster_id']:
            team_scores = weekly_scores_df[weekly_scores_df['roster_id'] == roster_id]['final_points'].values

            if len(team_scores) == 0:
                expected_wins.append({'roster_id': roster_id, 'expected_wins': 0, 'luck_factor': 0})
                continue

            # Calculate expected wins by comparing to all other teams each week
            total_expected_wins = 0

            for week_score in team_scores:
                if pd.isna(week_score) or week_score == 0:
                    continue

                # Get all other team scores for same weeks
                week = weekly_scores_df[
                    (weekly_scores_df['roster_id'] == roster_id) &
                    (weekly_scores_df['final_points'] == week_score)
                    ]['week'].iloc[0]

                other_scores = weekly_scores_df[
                    (weekly_scores_df['week'] == week) &
                    (weekly_scores_df['roster_id'] != roster_id)
                    ]['final_points'].values

                # Expected wins = percentage of teams we would beat
                if len(other_scores) > 0:
                    expected_wins_week = np.sum(week_score > other_scores) / len(other_scores)
                    total_expected_wins += expected_wins_week

            # Get actual wins through specified week
            actual_wins = records_df[records_df['roster_id'] == roster_id]['wins'].iloc[0]
            luck_factor = actual_wins - total_expected_wins

            expected_wins.append({
                'roster_id': roster_id,
                'expected_wins': total_expected_wins,
                'luck_factor': luck_factor
            })

        return pd.DataFrame(expected_wins)

    def calculate_momentum(self, weekly_scores_df, window=4, through_week=None):
        """Calculate recent performance momentum through specified week"""
        # Filter data if through_week is specified
        if through_week:
            weekly_scores_df = weekly_scores_df[weekly_scores_df['week'] <= through_week].copy()

        momentum_data = []

        for roster_id in weekly_scores_df['roster_id'].unique():
            team_scores = weekly_scores_df[
                weekly_scores_df['roster_id'] == roster_id
                ].sort_values('week')['final_points'].values

            if len(team_scores) >= window:
                recent_avg = np.mean(team_scores[-window:])
                season_avg = np.mean(team_scores)
                momentum = (recent_avg - season_avg) / season_avg if season_avg > 0 else 0
                recent_avg_val = recent_avg
            elif len(team_scores) > 0:
                # If we don't have enough weeks, use all available data
                recent_avg_val = np.mean(team_scores)
                season_avg = recent_avg_val
                momentum = 0
            else:
                recent_avg_val = 0
                momentum = 0

            momentum_data.append({
                'roster_id': roster_id,
                'momentum': momentum,
                'recent_avg': recent_avg_val
            })

        return pd.DataFrame(momentum_data)

    def calculate_positional_strength(self, player_stats_df, league_settings):
        """Calculate team strength by position"""
        if player_stats_df.empty:
            return pd.DataFrame()

        # Define position groups
        position_groups = {
            'QB': ['QB'],
            'RB': ['RB'],
            'WR': ['WR'],
            'TE': ['TE'],
            'K': ['K'],
            'DEF': ['DEF']
        }

        # Calculate positional rankings
        positional_strength = []

        for roster_id in player_stats_df['roster_id'].unique():
            team_players = player_stats_df[player_stats_df['roster_id'] == roster_id]
            team_strength = {'roster_id': roster_id}

            for pos_group, positions in position_groups.items():
                pos_players = team_players[team_players['position'].isin(positions)]

                if not pos_players.empty:
                    # Get top players at position (assuming starter requirements)
                    top_players = pos_players.nlargest(2, 'avg_points')
                    avg_points = top_players['avg_points'].mean()

                    # Calculate percentile vs all players at position
                    all_pos_players = player_stats_df[player_stats_df['position'].isin(positions)]
                    if not all_pos_players.empty:
                        percentile = (all_pos_players['avg_points'] < avg_points).mean() * 100
                        team_strength[f'{pos_group}_strength'] = percentile
                    else:
                        team_strength[f'{pos_group}_strength'] = 50
                else:
                    team_strength[f'{pos_group}_strength'] = 50

            # Overall positional strength (weighted average)
            weights = {'QB': 0.25, 'RB': 0.25, 'WR': 0.25, 'TE': 0.15, 'K': 0.05, 'DEF': 0.05}
            overall_strength = sum(
                team_strength.get(f'{pos}_strength', 50) * weight
                for pos, weight in weights.items()
            )
            team_strength['overall_positional_strength'] = overall_strength

            positional_strength.append(team_strength)

        return pd.DataFrame(positional_strength)

    def get_available_weeks(self, league_id):
        """Get list of weeks with available data"""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT DISTINCT week 
            FROM matchups 
            WHERE league_id = ? AND points > 0
            ORDER BY week
        """

        result = pd.read_sql_query(query, conn, params=(league_id,))
        conn.close()

        return result['week'].tolist() if not result.empty else []

    def create_odds_trend_chart(self, league_id, team_list):
        """Create a line chart showing odds progression through weeks"""
        available_weeks = self.get_available_weeks(league_id)

        if len(available_weeks) < 2:
            return None

        # Calculate odds for each week
        odds_progression = []

        for week in available_weeks:
            odds_df, _ = self.calculate_championship_odds(league_id, through_week=week)

            if not odds_df.empty:
                for _, team in odds_df.iterrows():
                    odds_progression.append({
                        'week': f"Week {week}",
                        'team_name': team['team_name'],
                        'championship_odds': team['championship_odds'],
                        'week_num': week
                    })

        if not odds_progression:
            return None

        progression_df = pd.DataFrame(odds_progression)

        # Filter to top teams to avoid clutter
        if team_list:
            progression_df = progression_df[progression_df['team_name'].isin(team_list)]

        # Create line chart
        fig = px.line(
            progression_df,
            x='week_num',
            y='championship_odds',
            color='team_name',
            title='Championship Odds Progression Throughout Season',
            labels={'week_num': 'Week', 'championship_odds': 'Championship Odds (%)', 'team_name': 'Team'}
        )

        fig.update_layout(
            height=500,
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            yaxis=dict(title='Championship Odds (%)'),
            hovermode='x unified'
        )

        return fig

    def calculate_championship_odds(self, league_id, through_week=None):
        """Calculate championship odds using multiple factors, optionally through a specific week"""
        # Get all data
        league_settings = self.get_league_settings(league_id)
        records_df = self.get_team_records_through_week(league_id, through_week)
        weekly_scores_df = self.get_weekly_scores(league_id)
        player_stats_df = self.get_player_performance(league_id)

        # Filter weekly scores if through_week specified
        if through_week:
            weekly_scores_df = weekly_scores_df[weekly_scores_df['week'] <= through_week].copy()

        if records_df.empty:
            return pd.DataFrame(), through_week

        # Calculate component metrics
        expected_wins_df = self.calculate_expected_wins(weekly_scores_df, records_df, through_week)
        momentum_df = self.calculate_momentum(weekly_scores_df, through_week=through_week)
        sos_df = self.calculate_strength_of_schedule(league_id, records_df, through_week)
        positional_strength_df = self.calculate_positional_strength(player_stats_df, league_settings)

        # Merge all data
        odds_df = records_df.copy()
        odds_df = odds_df.merge(expected_wins_df, on='roster_id', how='left')
        odds_df = odds_df.merge(momentum_df, on='roster_id', how='left')
        odds_df = odds_df.merge(sos_df, on='roster_id', how='left')
        odds_df = odds_df.merge(positional_strength_df, on='roster_id', how='left')

        # Fill missing values
        odds_df = odds_df.fillna(0)

        # Calculate composite score
        odds_df['win_pct'] = odds_df['wins'] / (odds_df['wins'] + odds_df['losses'] + odds_df['ties']).replace(0, 1)

        # Normalize factors (0-1 scale)
        factors = ['win_pct', 'points_for', 'expected_wins', 'momentum', 'overall_positional_strength',
                   'played_sos_overall']

        for factor in factors:
            if factor in odds_df.columns:
                col_min = odds_df[factor].min()
                col_max = odds_df[factor].max()
                if col_max > col_min:
                    odds_df[f'{factor}_norm'] = (odds_df[factor] - col_min) / (col_max - col_min)
                else:
                    odds_df[f'{factor}_norm'] = 0.5

        # Inverse normalize remaining SOS (lower remaining SOS is better)
        if 'remaining_sos' in odds_df.columns:
            col_min = odds_df['remaining_sos'].min()
            col_max = odds_df['remaining_sos'].max()
            if col_max > col_min:
                odds_df['remaining_sos_norm'] = 1 - (odds_df['remaining_sos'] - col_min) / (col_max - col_min)
            else:
                odds_df['remaining_sos_norm'] = 0.5

        # Weighted composite score - updated weights to include SOS
        if through_week and through_week <= 6:  # Early season - emphasize talent over record
            weights = {
                'win_pct_norm': 0.15,
                'points_for_norm': 0.25,
                'expected_wins_norm': 0.20,
                'momentum_norm': 0.10,
                'overall_positional_strength_norm': 0.20,
                'played_sos_overall_norm': 0.05,  # Played SOS
                'remaining_sos_norm': 0.05  # Remaining SOS
            }
        elif through_week and through_week >= 12:  # Late season - emphasize current form
            weights = {
                'win_pct_norm': 0.30,
                'points_for_norm': 0.15,
                'expected_wins_norm': 0.15,
                'momentum_norm': 0.20,
                'overall_positional_strength_norm': 0.10,
                'played_sos_overall_norm': 0.05,
                'remaining_sos_norm': 0.05
            }
        else:  # Mid-season or full season - balanced approach
            weights = {
                'win_pct_norm': 0.25,
                'points_for_norm': 0.20,
                'expected_wins_norm': 0.15,
                'momentum_norm': 0.15,
                'overall_positional_strength_norm': 0.15,
                'played_sos_overall_norm': 0.05,
                'remaining_sos_norm': 0.05
            }

        odds_df['composite_score'] = 0
        for factor, weight in weights.items():
            if factor in odds_df.columns:
                odds_df['composite_score'] += odds_df[factor] * weight

        # Convert to championship odds (percentage)
        total_score = odds_df['composite_score'].sum()
        if total_score > 0:
            odds_df['championship_odds'] = (odds_df['composite_score'] / total_score) * 100
        else:
            odds_df['championship_odds'] = 100 / len(odds_df)

        # Calculate betting odds (American format)
        odds_df['betting_odds'] = odds_df['championship_odds'].apply(self.convert_to_betting_odds)

        return odds_df.sort_values('championship_odds', ascending=False), through_week

    def convert_to_betting_odds(self, probability):
        """Convert percentage to American betting odds"""
        if probability >= 50:
            return f"-{int(100 * probability / (100 - probability))}"
        else:
            return f"+{int(100 * (100 - probability) / probability)}"

    def create_odds_visualization(self, odds_df):
        """Create visualization of championship odds"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Championship Odds', 'Win % vs Expected Win %',
                            'Points For vs Momentum', 'Positional Strength'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # Championship odds bar chart
        fig.add_trace(
            go.Bar(
                x=odds_df['team_name'],
                y=odds_df['championship_odds'],
                name='Championship Odds (%)',
                text=[f"{x:.1f}%" for x in odds_df['championship_odds']],
                textposition='auto',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # Win % vs Expected Win % scatter
        fig.add_trace(
            go.Scatter(
                x=odds_df['expected_wins'],
                y=odds_df['wins'],
                mode='markers+text',
                text=odds_df['team_name'],
                textposition="top center",
                name='Actual vs Expected Wins',
                marker=dict(size=10, color='red')
            ),
            row=1, col=2
        )

        # Points vs Momentum scatter
        fig.add_trace(
            go.Scatter(
                x=odds_df['points_for'],
                y=odds_df['momentum'],
                mode='markers+text',
                text=odds_df['team_name'],
                textposition="top center",
                name='Points vs Momentum',
                marker=dict(size=10, color='green')
            ),
            row=2, col=1
        )

        # Positional strength bar
        fig.add_trace(
            go.Bar(
                x=odds_df['team_name'],
                y=odds_df['overall_positional_strength'],
                name='Positional Strength',
                text=[f"{x:.0f}" for x in odds_df['overall_positional_strength']],
                textposition='auto',
                marker_color='orange'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Championship Contender Analytics Dashboard"
        )

        # Update axes labels
        fig.update_xaxes(title_text="Team", row=1, col=1)
        fig.update_yaxes(title_text="Odds (%)", row=1, col=1)
        fig.update_xaxes(title_text="Expected Wins", row=1, col=2)
        fig.update_yaxes(title_text="Actual Wins", row=1, col=2)
        fig.update_xaxes(title_text="Points For", row=2, col=1)
        fig.update_yaxes(title_text="Momentum", row=2, col=1)
        fig.update_xaxes(title_text="Team", row=2, col=2)
        fig.update_yaxes(title_text="Strength Percentile", row=2, col=2)

        return fig

    def render(self, league_id, season):
        """Render the championship odds module"""
        st.header("🏆 KLYT Picks (Courtesy of Klytics v3 Simulation Model)")
        st.markdown("Advanced analytics to determine championship odds based on multiple performance factors")

        # Get available weeks for filtering
        available_weeks = self.get_available_weeks(league_id)

        if not available_weeks:
            st.warning("No matchup data available for championship odds calculation")
            return

        # Week filter control
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Week slider
            selected_week = st.select_slider(
                "📅 View Odds Through Week:",
                options=available_weeks + ["Current"],
                value="Current",
                help="See how championship odds looked at different points in the season"
            )

        with col2:
            show_trends = st.checkbox(
                "📈 Show Trends",
                value=False,
                help="Display odds progression chart"
            )

        with col3:
            if st.button("🔄 Refresh", help="Recalculate odds"):
                st.rerun()

        # Calculate odds
        through_week = None if selected_week == "Current" else selected_week

        with st.spinner(
                f"Calculating championship odds{'through week ' + str(selected_week) if through_week else ''}..."):
            odds_df, analysis_week = self.calculate_championship_odds(league_id, through_week)

        if odds_df.empty:
            st.warning("No data available for championship odds calculation")
            return

        # Display context
        if analysis_week:
            st.info(f"📊 **Analysis Period**: Through Week {analysis_week} of the {season} season")
        else:
            st.info(f"📊 **Analysis Period**: Full {season} season (current standings)")

        # Show trends chart if requested
        if show_trends and len(available_weeks) >= 2:
            st.subheader("📈 Championship Odds Progression")

            # Get top 6 teams for trend display
            top_teams = odds_df.head(6)['team_name'].tolist()

            trend_fig = self.create_odds_trend_chart(league_id, top_teams)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.warning("Unable to generate trends chart - insufficient data")

            st.markdown("---")

        # Display top contenders
        st.subheader(f"🥇 Championship Contenders{' (Through Week ' + str(selected_week) + ')' if through_week else ''}")

        # Top 6 contenders
        top_contenders = odds_df.head(6)

        cols = st.columns(3)
        for i, (_, team) in enumerate(top_contenders.iterrows()):
            col_idx = i % 3
            with cols[col_idx]:
                # Medal emojis for top 3
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i + 1}"

                # Show actual wins from the analysis period
                wins_display = team.get('actual_wins', team['wins'])

                st.metric(
                    label=f"{medal} {team['team_name']}",
                    value=f"{team['championship_odds']:.1f}%",
                    delta=f"{team['betting_odds']}",
                    help=f"Record: {wins_display}-{team['losses']}, Points: {team['points_for']:.1f}"
                )

        # Detailed table
        st.subheader("📊 Detailed Championship Odds")

        display_df = odds_df[[
            'team_name', 'wins', 'losses', 'win_pct', 'points_for', 'expected_wins',
            'momentum', 'played_sos_overall', 'remaining_sos', 'overall_positional_strength',
            'championship_odds', 'betting_odds'
        ]].copy()

        # Use actual_wins if available (for filtered analysis)
        if 'actual_wins' in odds_df.columns:
            display_df['wins'] = odds_df['actual_wins']

        # Format columns
        display_df['win_pct'] = display_df['win_pct'].apply(lambda x: f"{x:.3f}")
        display_df['points_for'] = display_df['points_for'].apply(lambda x: f"{x:.1f}")
        display_df['expected_wins'] = display_df['expected_wins'].apply(lambda x: f"{x:.1f}")
        display_df['momentum'] = display_df['momentum'].apply(lambda x: f"{x:+.3f}")
        display_df['played_sos_overall'] = display_df['played_sos_overall'].apply(lambda x: f"{x:.3f}")
        display_df['remaining_sos'] = display_df['remaining_sos'].apply(lambda x: f"{x:.3f}")
        display_df['overall_positional_strength'] = display_df['overall_positional_strength'].apply(
            lambda x: f"{x:.0f}")
        display_df['championship_odds'] = display_df['championship_odds'].apply(lambda x: f"{x:.1f}%")

        # Rename columns for display
        display_df.columns = [
            'Team', 'Wins', 'Losses', 'Win %', 'Points For', 'Expected Wins',
            'Momentum', 'Played SOS', 'Remaining SOS', 'Positional Strength', 'Championship Odds', 'Betting Odds'
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # Visualizations
        st.subheader("📈 Analytics Dashboard")
        fig = self.create_odds_visualization(odds_df)
        st.plotly_chart(fig, use_container_width=True)

        # Key insights
        st.subheader("🔍 Key Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🏆 Championship Favorite**")
            favorite = odds_df.iloc[0]
            wins_display = favorite.get('actual_wins', favorite['wins'])
            st.info(f"""
            **{favorite['team_name']}** leads with **{favorite['championship_odds']:.1f}%** odds
            - Record: {wins_display}-{favorite['losses']} ({favorite['win_pct']:.3f})
            - Points For: {favorite['points_for']:.1f}
            - Expected Wins: {favorite['expected_wins']:.1f}
            - Betting Odds: {favorite['betting_odds']}
            """)

            st.markdown("**🔥 Hot Hand (Best Momentum)**")
            hot_team = odds_df.nlargest(1, 'momentum').iloc[0]
            st.success(f"""
            **{hot_team['team_name']}** is trending up
            - Momentum: {hot_team['momentum']:+.3f}
            - Recent performance above season average
            """)

            # Strength of schedule insights
            if 'played_sos_overall' in odds_df.columns:
                st.markdown("**💪 Toughest Schedule Survived**")
                toughest_sos = odds_df.nlargest(1, 'played_sos_overall').iloc[0]
                st.warning(f"""
                **{toughest_sos['team_name']}** faced the hardest opponents
                - Played SOS: {toughest_sos['played_sos_overall']:.3f}
                - Championship Odds: {toughest_sos['championship_odds']:.1f}%
                - May be battle-tested for playoffs
                """)

        with col2:
            st.markdown("**🎯 Best Value Bet**")
            # Find team with biggest positive difference between expected and actual wins
            value_bet = odds_df.nlargest(1, 'expected_wins').iloc[0]
            st.warning(f"""
            **{value_bet['team_name']}** might be undervalued
            - Championship Odds: {value_bet['championship_odds']:.1f}%
            - Expected Wins: {value_bet['expected_wins']:.1f}
            - Actual Wins: {value_bet.get('actual_wins', value_bet['wins'])}
            """)

            st.markdown("**🛡️ Strongest Roster**")
            strongest = odds_df.nlargest(1, 'overall_positional_strength').iloc[0]
            st.info(f"""
            **{strongest['team_name']}** has the best players
            - Positional Strength: {strongest['overall_positional_strength']:.0f}th percentile
            - Championship Odds: {strongest['championship_odds']:.1f}%
            """)

            # Easy remaining schedule
            if 'remaining_sos' in odds_df.columns and odds_df['remaining_sos'].max() > 0:
                st.markdown("**📈 Easiest Path Ahead**")
                easiest_remaining = odds_df.nsmallest(1, 'remaining_sos').iloc[0]
                st.success(f"""
                **{easiest_remaining['team_name']}** has the easiest remaining schedule
                - Remaining SOS: {easiest_remaining['remaining_sos']:.3f}
                - Championship Odds: {easiest_remaining['championship_odds']:.1f}%
                - Favorable schedule could boost playoff chances
                """)

        # Week-specific insights
        if through_week:
            st.subheader(f"📅 Week {through_week} Historical Context")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔮 Season Stage Analysis**")
                if through_week <= 4:
                    st.info(f"""
                    **Very Early Season Analysis (Week {through_week})**
                    - Small sample sizes - results may be volatile
                    - Positional strength weighted more heavily
                    - Focus on talent over small sample results
                    - Schedule strength less reliable
                    """)
                elif through_week <= 8:
                    st.info(f"""
                    **Mid-Season Analysis (Week {through_week})**
                    - Sample sizes becoming more reliable
                    - Balanced weighting of all factors
                    - Momentum trends starting to matter
                    - Schedule strength gaining relevance
                    """)
                else:
                    st.info(f"""
                    **Late Season Analysis (Week {through_week})**
                    - Win percentage weighted more heavily
                    - Momentum and recent form crucial
                    - Schedule strength very important
                    - Playoff push implications
                    """)

            with col2:
                st.markdown("**📊 Analysis Methodology**")
                if through_week <= 6:
                    st.warning(f"""
                    **Early Season Weighting (Week {through_week}):**
                    - Win % (15%) - Small sample, less reliable
                    - Points For (25%) - More emphasis on scoring
                    - Expected Wins (20%) - Key early indicator
                    - Positional Strength (20%) - Talent matters most
                    - Momentum (10%) - Limited data
                    - SOS (10%) - Less reliable early
                    """)
                elif through_week >= 12:
                    st.warning(f"""
                    **Late Season Weighting (Week {through_week}):**
                    - Win % (30%) - Record matters most
                    - Points For (15%) - Still important
                    - Expected Wins (15%) - Solid indicator
                    - Momentum (20%) - Hot teams crucial
                    - Positional Strength (10%) - Known quantity
                    - SOS (10%) - Critical for playoffs
                    """)
                else:
                    st.warning(f"""
                    **Mid-Season Weighting (Week {through_week}):**
                    - Win % (25%) - Standard importance
                    - Points For (20%) - Key indicator
                    - Expected Wins (15%) - Reliable metric
                    - Momentum (15%) - Growing importance
                    - Positional Strength (15%) - Known factor
                    - SOS (10%) - Gaining relevance
                    """)

        # Strength of Schedule Deep Dive
        if 'played_sos_overall' in odds_df.columns and 'remaining_sos' in odds_df.columns:
            st.subheader("🗓️ Strength of Schedule Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🏃‍♂️ Hardest Road Traveled**")
                sos_df = odds_df.nlargest(3, 'played_sos_overall')[
                    ['team_name', 'played_sos_overall', 'wins', 'championship_odds']]
                for _, team in sos_df.iterrows():
                    wins_display = team.get('actual_wins', team['wins'])
                    st.metric(
                        team['team_name'],
                        f"{team['played_sos_overall']:.3f}",
                        f"{wins_display} wins ({team['championship_odds']:.1f}%)"
                    )

            with col2:
                st.markdown("**🛤️ Easiest Path Ahead**")
                easy_df = odds_df.nsmallest(3, 'remaining_sos')[['team_name', 'remaining_sos', 'championship_odds']]
                for _, team in easy_df.iterrows():
                    if team['remaining_sos'] > 0:  # Only show if remaining games exist
                        st.metric(
                            team['team_name'],
                            f"{team['remaining_sos']:.3f}",
                            f"{team['championship_odds']:.1f}% odds"
                        )

            with col3:
                st.markdown("**⚖️ Schedule Balance**")
                # Calculate schedule variance (difference between played and remaining)
                if not odds_df.empty:
                    odds_df_copy = odds_df.copy()
                    odds_df_copy['sos_variance'] = odds_df_copy['remaining_sos'] - odds_df_copy['played_sos_overall']
                    odds_df_copy['abs_sos_variance'] = abs(odds_df_copy['sos_variance'])
                    balanced_df = odds_df_copy.nsmallest(3, 'abs_sos_variance')[
                        ['team_name', 'sos_variance', 'championship_odds']]
                    for _, team in balanced_df.iterrows():
                        variance_text = "Easier ahead" if team['sos_variance'] < 0 else "Harder ahead" if team[
                                                                                                              'sos_variance'] > 0 else "Balanced"
                        st.metric(
                            team['team_name'],
                            variance_text,
                            f"{team['championship_odds']:.1f}% odds"
                        )

        # Methodology explanation
        with st.expander("📚 How Championship Odds Are Calculated"):
            st.markdown(f"""
            The championship odds are calculated using a weighted composite score that adapts based on the season stage:

            **Core Factors:**
            - **Win Percentage**: Current record and winning percentage{' through Week ' + str(through_week) if through_week else ''}
            - **Points Scored**: Total fantasy points{' through Week ' + str(through_week) if through_week else ''} - indicates offensive capability  
            - **Expected Wins**: Wins based on scoring vs opponents each week (accounts for luck)
            - **Momentum**: Recent 4-week performance vs season average{' (or available weeks)' if through_week else ''}
            - **Positional Strength**: Player rankings by position vs league (percentile-based)
            - **Strength of Schedule**: Both played opponents and remaining schedule difficulty

            **Strength of Schedule Components:**
            - **Played SOS**: Average win rate and scoring of opponents faced (60% record, 40% points)
            - **Remaining SOS**: Average win rate of future opponents
            - **Overall Impact**: Higher played SOS may indicate battle-tested teams; lower remaining SOS suggests easier path

            **Adaptive Weighting by Season Stage:**
            {"- **Early Season (≤Week 6)**: Emphasizes talent/points over small-sample records" if through_week and through_week <= 6 else ""}
            {"- **Mid Season (Week 7-11)**: Balanced approach across all factors" if through_week and 7 <= through_week <= 11 else ""}
            {"- **Late Season (≥Week 12)**: Emphasizes current record and momentum" if through_week and through_week >= 12 else ""}
            {"- **Full Season**: Balanced weighting across all factors" if not through_week else ""}

            **Additional Metrics:**
            - **Luck Factor**: Difference between actual and expected wins
            - **Betting Odds**: American format odds based on championship probability
            - **SOS Variance**: Difference between remaining and played schedule difficulty

            {'**Week Filter Impact**: When viewing through a specific week, all stats are recalculated using only data available at that point.' if through_week else ''}

            *Note: Odds are for entertainment purposes and based on historical performance*
            """)

            # Show current calculation weights
            st.markdown("**Current Composite Score Weights:**")
            if through_week and through_week <= 6:
                weights_data = [
                    ['Win Percentage', '15%', 'Small sample, less emphasis'],
                    ['Points Scored', '25%', 'Key early indicator of team strength'],
                    ['Expected Wins', '20%', 'Accounts for luck in small samples'],
                    ['Recent Momentum', '10%', 'Limited data available'],
                    ['Positional Strength', '20%', 'Talent evaluation crucial early'],
                    ['Played SOS', '5%', 'Limited sample of opponents'],
                    ['Remaining SOS', '5%', 'Future schedule preview']
                ]
            elif through_week and through_week >= 12:
                weights_data = [
                    ['Win Percentage', '30%', 'Record most important late season'],
                    ['Points Scored', '15%', 'Still relevant but less weight'],
                    ['Expected Wins', '15%', 'Solid performance indicator'],
                    ['Recent Momentum', '20%', 'Hot teams crucial for playoffs'],
                    ['Positional Strength', '10%', 'Known quantity by now'],
                    ['Played SOS', '5%', 'Historical context'],
                    ['Remaining SOS', '5%', 'Critical for playoff push']
                ]
            else:
                weights_data = [
                    ['Win Percentage', '25%', 'Balanced record importance'],
                    ['Points Scored', '20%', 'Key offensive indicator'],
                    ['Expected Wins', '15%', 'Performance vs schedule'],
                    ['Recent Momentum', '15%', 'Form and trends'],
                    ['Positional Strength', '15%', 'Player talent evaluation'],
                    ['Played SOS', '5%', 'Opponent difficulty faced'],
                    ['Remaining SOS', '5%', 'Schedule difficulty ahead']
                ]

            weights_df = pd.DataFrame(weights_data, columns=['Factor', 'Weight', 'Description'])
            st.dataframe(weights_df, hide_index=True, use_container_width=True)