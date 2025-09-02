import sqlite3
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sleeper_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SleeperDataPipeline:
    def __init__(self, username: str, db_path: str = "sleeper_data.db", max_workers: int = 10):
        self.username = username
        self.db_path = db_path
        self.max_workers = max_workers
        self.base_url = "https://api.sleeper.app/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SleeperDataPipeline/1.0',
            'Accept': 'application/json'
        })

        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with all necessary tables"""
        logger.info("Initializing database...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                display_name TEXT,
                avatar TEXT,
                created_at INTEGER,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Leagues table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leagues (
                league_id TEXT PRIMARY KEY,
                name TEXT,
                season TEXT,
                total_rosters INTEGER,
                status TEXT,
                sport TEXT,
                scoring_settings TEXT, -- JSON
                roster_positions TEXT, -- JSON
                playoff_week_start INTEGER,
                trade_deadline INTEGER,
                waiver_type TEXT,
                draft_settings TEXT, -- JSON
                settings TEXT, -- JSON
                created_at INTEGER,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # League users (many-to-many relationship)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS league_users (
                league_id TEXT,
                user_id TEXT,
                is_owner BOOLEAN,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (league_id, user_id),
                FOREIGN KEY (league_id) REFERENCES leagues(league_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')

        # Rosters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rosters (
                roster_id INTEGER,
                league_id TEXT,
                owner_id TEXT,
                co_owners TEXT, -- JSON array
                players TEXT, -- JSON array
                reserve TEXT, -- JSON array
                taxi TEXT, -- JSON array
                starters TEXT, -- JSON array
                settings TEXT, -- JSON (wins, losses, fpts, etc.)
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (roster_id, league_id),
                FOREIGN KEY (league_id) REFERENCES leagues(league_id),
                FOREIGN KEY (owner_id) REFERENCES users(user_id)
            )
        ''')

        # Matchups table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matchups (
                matchup_id INTEGER,
                league_id TEXT,
                week INTEGER,
                roster_id INTEGER,
                points REAL,
                points_decimal REAL,
                starters TEXT, -- JSON array
                starters_points TEXT, -- JSON array
                players_points TEXT, -- JSON
                custom_points REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (league_id, week, roster_id),
                FOREIGN KEY (league_id) REFERENCES leagues(league_id)
            )
        ''')

        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                league_id TEXT,
                type TEXT, -- trade, waiver, free_agent
                status TEXT,
                creator TEXT,
                created INTEGER,
                roster_ids TEXT, -- JSON array
                consenter_ids TEXT, -- JSON array
                adds TEXT, -- JSON
                drops TEXT, -- JSON
                draft_picks TEXT, -- JSON array
                waiver_budget TEXT, -- JSON array
                settings TEXT, -- JSON
                metadata TEXT, -- JSON
                leg INTEGER,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (league_id) REFERENCES leagues(league_id)
            )
        ''')

        # Players table (NFL players info)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                full_name TEXT,
                first_name TEXT,
                last_name TEXT,
                position TEXT,
                team TEXT,
                college TEXT,
                height TEXT,
                weight TEXT,
                age INTEGER,
                years_exp INTEGER,
                status TEXT,
                injury_status TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Data refresh log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS refresh_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT,
                league_id TEXT,
                season TEXT,
                records_updated INTEGER,
                started_at DATETIME,
                completed_at DATETIME,
                success BOOLEAN,
                error_message TEXT
            )
        ''')

        # Create indexes for better query performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_leagues_season ON leagues(season)",
            "CREATE INDEX IF NOT EXISTS idx_rosters_league ON rosters(league_id)",
            "CREATE INDEX IF NOT EXISTS idx_matchups_league_week ON matchups(league_id, week)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_league ON transactions(league_id)",
            "CREATE INDEX IF NOT EXISTS idx_league_users_user ON league_users(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_players_position ON players(position)",
            "CREATE INDEX IF NOT EXISTS idx_players_team ON players(team)"
        ]

        for index in indexes:
            cursor.execute(index)

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def make_request(self, url: str, timeout: int = 30) -> Optional[dict]:
        """Make API request with error handling"""
        try:
            response = self.session.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                logger.warning(f"Rate limited, waiting 60 seconds...")
                time.sleep(60)
                return self.make_request(url, timeout)
            else:
                logger.warning(f"API returned {response.status_code} for {url}")
                return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def batch_requests(self, urls: List[str]) -> Dict[str, dict]:
        """Make multiple requests concurrently"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.make_request, url): url for url in urls}

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results[url] = result
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    results[url] = None

                # Rate limiting protection
                time.sleep(0.1)

        return results

    def get_user_info(self) -> Optional[dict]:
        """Get user information"""
        logger.info(f"Fetching user info for {self.username}")
        return self.make_request(f"{self.base_url}/user/{self.username}")

    def get_user_leagues(self, user_id: str, seasons: List[str]) -> Dict[str, List[dict]]:
        """Get user leagues for multiple seasons"""
        logger.info(f"Fetching leagues for seasons: {seasons}")

        urls = [f"{self.base_url}/user/{user_id}/leagues/nfl/{season}" for season in seasons]
        results = self.batch_requests(urls)

        leagues_by_season = {}
        for i, season in enumerate(seasons):
            url = urls[i]
            leagues_by_season[season] = results.get(url, []) or []

        return leagues_by_season

    def get_league_data(self, league_ids: List[str]) -> Dict[str, dict]:
        """Get detailed data for multiple leagues"""
        logger.info(f"Fetching detailed data for {len(league_ids)} leagues")

        all_urls = []
        url_mapping = {}

        for league_id in league_ids:
            urls = [
                f"{self.base_url}/league/{league_id}",
                f"{self.base_url}/league/{league_id}/users",
                f"{self.base_url}/league/{league_id}/rosters"
            ]
            all_urls.extend(urls)
            url_mapping[league_id] = {
                'info': urls[0],
                'users': urls[1],
                'rosters': urls[2]
            }

        results = self.batch_requests(all_urls)

        league_data = {}
        for league_id, urls in url_mapping.items():
            league_data[league_id] = {
                'info': results.get(urls['info']),
                'users': results.get(urls['users'], []),
                'rosters': results.get(urls['rosters'], [])
            }

        return league_data

    def get_all_matchups(self, league_ids: List[str], weeks: List[int]) -> Dict[str, Dict[int, List[dict]]]:
        """Get matchups for multiple leagues and weeks"""
        logger.info(f"Fetching matchups for {len(league_ids)} leagues, weeks {min(weeks)}-{max(weeks)}")

        all_urls = []
        url_mapping = {}

        for league_id in league_ids:
            url_mapping[league_id] = {}
            for week in weeks:
                url = f"{self.base_url}/league/{league_id}/matchups/{week}"
                all_urls.append(url)
                url_mapping[league_id][week] = url

        results = self.batch_requests(all_urls)

        matchup_data = {}
        for league_id in league_ids:
            matchup_data[league_id] = {}
            for week in weeks:
                url = url_mapping[league_id][week]
                matchup_data[league_id][week] = results.get(url, []) or []

        return matchup_data

    def get_all_transactions(self, league_ids: List[str], weeks: List[int]) -> Dict[str, List[dict]]:
        """Get transactions for multiple leagues"""
        logger.info(f"Fetching transactions for {len(league_ids)} leagues")

        all_urls = []
        url_mapping = {}

        for league_id in league_ids:
            url_mapping[league_id] = []
            for week in weeks:
                url = f"{self.base_url}/league/{league_id}/transactions/{week}"
                all_urls.append(url)
                url_mapping[league_id].append(url)

        results = self.batch_requests(all_urls)

        transaction_data = {}
        for league_id in league_ids:
            all_transactions = []
            for url in url_mapping[league_id]:
                week_transactions = results.get(url, []) or []
                all_transactions.extend(week_transactions)
            transaction_data[league_id] = all_transactions

        return transaction_data

    def get_nfl_players(self) -> Optional[dict]:
        """Get all NFL players data"""
        logger.info("Fetching NFL players data")
        return self.make_request(f"{self.base_url}/players/nfl")

    def store_users(self, users_data: List[dict]):
        """Store users data in database"""
        if not users_data:
            return

        logger.info(f"Storing {len(users_data)} users")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for user in users_data:
            if not user:
                continue

            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, username, display_name, avatar, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                user.get('user_id'),
                user.get('username'),
                user.get('display_name'),
                user.get('avatar'),
                user.get('created')
            ))

        conn.commit()
        conn.close()

    def store_leagues(self, leagues_data: Dict[str, List[dict]]):
        """Store leagues data in database"""
        logger.info("Storing leagues data")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for season, leagues in leagues_data.items():
            for league in leagues:
                if not league:
                    continue

                cursor.execute('''
                    INSERT OR REPLACE INTO leagues 
                    (league_id, name, season, total_rosters, status, sport, scoring_settings,
                     roster_positions, playoff_week_start, trade_deadline, waiver_type,
                     draft_settings, settings, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    league.get('league_id'),
                    league.get('name'),
                    league.get('season'),
                    league.get('total_rosters'),
                    league.get('status'),
                    league.get('sport'),
                    json.dumps(league.get('scoring_settings', {})),
                    json.dumps(league.get('roster_positions', [])),
                    league.get('playoff_week_start'),
                    league.get('trade_deadline'),
                    league.get('waiver_type'),
                    json.dumps(league.get('draft_settings', {})),
                    json.dumps(league.get('settings', {})),
                    league.get('created')
                ))

        conn.commit()
        conn.close()

    def store_league_data(self, league_data: Dict[str, dict]):
        """Store detailed league data (users, rosters)"""
        logger.info("Storing detailed league data")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for league_id, data in league_data.items():
            users = data.get('users', [])
            rosters = data.get('rosters', [])

            # Store league-user relationships
            for user in users:
                if not user:
                    continue

                cursor.execute('''
                    INSERT OR REPLACE INTO league_users (league_id, user_id, is_owner, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (league_id, user.get('user_id'), user.get('is_owner', False)))

            # Store rosters
            for roster in rosters:
                if not roster:
                    continue

                cursor.execute('''
                    INSERT OR REPLACE INTO rosters 
                    (roster_id, league_id, owner_id, co_owners, players, reserve, taxi,
                     starters, settings, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    roster.get('roster_id'),
                    league_id,
                    roster.get('owner_id'),
                    json.dumps(roster.get('co_owners', [])),
                    json.dumps(roster.get('players', [])),
                    json.dumps(roster.get('reserve', [])),
                    json.dumps(roster.get('taxi', [])),
                    json.dumps(roster.get('starters', [])),
                    json.dumps(roster.get('settings', {}))
                ))

        conn.commit()
        conn.close()

    def store_matchups(self, matchup_data: Dict[str, Dict[int, List[dict]]]):
        """Store matchups data"""
        logger.info("Storing matchups data")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for league_id, weeks_data in matchup_data.items():
            for week, matchups in weeks_data.items():
                for matchup in matchups:
                    if not matchup:
                        continue

                    cursor.execute('''
                        INSERT OR REPLACE INTO matchups 
                        (matchup_id, league_id, week, roster_id, points, points_decimal,
                         starters, starters_points, players_points, custom_points, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (
                        matchup.get('matchup_id'),
                        league_id,
                        week,
                        matchup.get('roster_id'),
                        matchup.get('points'),
                        matchup.get('points_decimal'),
                        json.dumps(matchup.get('starters', [])),
                        json.dumps(matchup.get('starters_points', [])),
                        json.dumps(matchup.get('players_points', {})),
                        matchup.get('custom_points')
                    ))

        conn.commit()
        conn.close()

    def store_transactions(self, transaction_data: Dict[str, List[dict]]):
        """Store transactions data"""
        logger.info("Storing transactions data")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for league_id, transactions in transaction_data.items():
            for transaction in transactions:
                if not transaction:
                    continue

                cursor.execute('''
                    INSERT OR REPLACE INTO transactions 
                    (transaction_id, league_id, type, status, creator, created, roster_ids,
                     consenter_ids, adds, drops, draft_picks, waiver_budget, settings,
                     metadata, leg, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    transaction.get('transaction_id'),
                    league_id,
                    transaction.get('type'),
                    transaction.get('status'),
                    transaction.get('creator'),
                    transaction.get('created'),
                    json.dumps(transaction.get('roster_ids', [])),
                    json.dumps(transaction.get('consenter_ids', [])),
                    json.dumps(transaction.get('adds', {})),
                    json.dumps(transaction.get('drops', {})),
                    json.dumps(transaction.get('draft_picks', [])),
                    json.dumps(transaction.get('waiver_budget', [])),
                    json.dumps(transaction.get('settings', {})),
                    json.dumps(transaction.get('metadata', {})),
                    transaction.get('leg')
                ))

        conn.commit()
        conn.close()

    def store_players(self, players_data: dict):
        """Store NFL players data"""
        if not players_data:
            return

        logger.info(f"Storing {len(players_data)} NFL players")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for player_id, player in players_data.items():
            cursor.execute('''
                INSERT OR REPLACE INTO players 
                (player_id, full_name, first_name, last_name, position, team, college,
                 height, weight, age, years_exp, status, injury_status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                player_id,
                player.get('full_name'),
                player.get('first_name'),
                player.get('last_name'),
                player.get('position'),
                player.get('team'),
                player.get('college'),
                player.get('height'),
                player.get('weight'),
                player.get('age'),
                player.get('years_exp'),
                player.get('status'),
                player.get('injury_status')
            ))

        conn.commit()
        conn.close()

    def log_refresh(self, table_name: str, league_id: str = None, season: str = None,
                    records_updated: int = 0, success: bool = True, error_message: str = None):
        """Log data refresh activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO refresh_log 
            (table_name, league_id, season, records_updated, started_at, completed_at, success, error_message)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
        ''', (table_name, league_id, season, records_updated, success, error_message))

        conn.commit()
        conn.close()

    def run_full_sync(self, seasons: List[str] = None, include_matchups: bool = True,
                      include_transactions: bool = True, include_players: bool = True):
        """Run complete data synchronization"""
        start_time = datetime.now()
        logger.info(f"Starting full sync at {start_time}")

        if not seasons:
            current_year = datetime.now().year
            seasons = [str(year) for year in range(current_year, current_year - 3, -1)]

        try:
            # 1. Get user info
            user_info = self.get_user_info()
            if not user_info:
                raise Exception("Could not fetch user info")

            user_id = user_info['user_id']
            self.store_users([user_info])

            # 2. Get all leagues
            leagues_data = self.get_user_leagues(user_id, seasons)
            self.store_leagues(leagues_data)

            # 3. Get all league IDs
            all_league_ids = []
            for season_leagues in leagues_data.values():
                all_league_ids.extend([league['league_id'] for league in season_leagues if league])

            logger.info(f"Found {len(all_league_ids)} leagues total")

            if not all_league_ids:
                logger.warning("No leagues found")
                return

            # 4. Get detailed league data
            league_data = self.get_league_data(all_league_ids)

            # Extract and store all users from all leagues
            all_users = []
            for data in league_data.values():
                all_users.extend(data.get('users', []))

            unique_users = {user['user_id']: user for user in all_users if user}.values()
            self.store_users(list(unique_users))
            self.store_league_data(league_data)

            # 5. Get matchups (optional)
            if include_matchups:
                weeks = list(range(1, 19))  # Weeks 1-18
                matchup_data = self.get_all_matchups(all_league_ids, weeks)
                self.store_matchups(matchup_data)

            # 6. Get transactions (optional)
            if include_transactions:
                weeks = list(range(1, 19))
                transaction_data = self.get_all_transactions(all_league_ids, weeks)
                self.store_transactions(transaction_data)

            # 7. Get NFL players (optional)
            if include_players:
                players_data = self.get_nfl_players()
                if players_data:
                    self.store_players(players_data)

            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Full sync completed in {duration}")

            # Log success
            self.log_refresh("full_sync", records_updated=len(all_league_ids), success=True)

        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            self.log_refresh("full_sync", success=False, error_message=str(e))
            raise

    def get_db_stats(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        tables = ['users', 'leagues', 'league_users', 'rosters', 'matchups', 'transactions', 'players']
        stats = {}

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        conn.close()
        return stats


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Sleeper Fantasy Football Data Pipeline')
    parser.add_argument('--username', default='maxitopapito', help='Sleeper username')
    parser.add_argument('--db-path', default='sleeper_data.db', help='SQLite database path')
    parser.add_argument('--seasons', nargs='+', help='Seasons to sync (e.g., 2024 2023)')
    parser.add_argument('--no-matchups', action='store_true', help='Skip matchups sync')
    parser.add_argument('--no-transactions', action='store_true', help='Skip transactions sync')
    parser.add_argument('--no-players', action='store_true', help='Skip players sync')
    parser.add_argument('--max-workers', type=int, default=10, help='Max concurrent requests')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SleeperDataPipeline(
        username=args.username,
        db_path=args.db_path,
        max_workers=args.max_workers
    )

    # Run sync
    try:
        pipeline.run_full_sync(
            seasons=args.seasons,
            include_matchups=not args.no_matchups,
            include_transactions=not args.no_transactions,
            include_players=not args.no_players
        )

        # Show stats
        stats = pipeline.get_db_stats()
        print("\n" + "=" * 50)
        print("DATABASE STATISTICS")
        print("=" * 50)
        for table, count in stats.items():
            print(f"{table:15}: {count:,} records")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()