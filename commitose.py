#!/usr/bin/env python3
"""
Commitose: Markov chain-based commit history simulator
"""

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict


@dataclass
class Commit:
    """A single commit with timestamp and message."""

    timestamp: datetime
    message: str


@dataclass
class Break:
    """A time period with scaled commit activity."""

    start: datetime
    end: datetime
    factor: float


@dataclass
class Config:
    """Configuration for commit generation."""

    # Time range
    start_date: datetime
    end_date: datetime
    timezone: str

    # Activity patterns
    weekly_weights: List[float]  # Mon-Sun activity multipliers
    breaks: List[Break]  # Time periods with scaled activity

    # Markov state machine
    transition_matrix: List[List[float]]  # State transitions
    state_means: List[float]  # Expected commits per state
    state_dispersion: float  # Variance parameter
    state_zero_inflation: List[float]  # Probability of zero commits

    # Intra-day timing
    session_windows: List[Tuple[int, int]]  # (hour, minute) for session starts
    session_std_minutes: float  # Jitter around session times
    session_count_distribution: List[List[float]]  # Sessions per state
    min_gap_seconds: int  # Between commits
    max_gap_seconds: int

    # Git settings
    user_name: str
    user_email: str
    repo_path: Path
    branch: str

    # Runtime options
    random_seed: Optional[int]
    dry_run: bool

    def validate(self) -> None:
        """Validate all configuration parameters."""
        errors = []

        if self.start_date > self.end_date:
            errors.append("start_date must be <= end_date")

        if len(self.weekly_weights) != 7:
            errors.append(
                f"weekly_weights needs 7 values, got {len(self.weekly_weights)}"
            )

        # Validate breaks
        for i, brk in enumerate(self.breaks):
            if brk.start > brk.end:
                errors.append(f"Break {i + 1}: start must be <= end")
            if brk.factor < 0:
                errors.append(f"Break {i + 1}: factor must be >= 0")

        # Check for overlapping breaks
        for i in range(len(self.breaks)):
            for j in range(i + 1, len(self.breaks)):
                brk1 = self.breaks[i]
                brk2 = self.breaks[j]
                # Check if date ranges overlap
                if (
                    brk1.start.date() <= brk2.end.date()
                    and brk1.end.date() >= brk2.start.date()
                ):
                    errors.append(
                        f"Breaks overlap: {brk1.start.date()} to {brk1.end.date()} "
                        f"and {brk2.start.date()} to {brk2.end.date()}"
                    )

        if not self._validate_matrix(self.transition_matrix, 4, 4):
            errors.append("transition_matrix must be 4x4 with rows summing to 1")

        if len(self.state_means) != 4:
            errors.append(f"state_means needs 4 values, got {len(self.state_means)}")

        if len(self.state_zero_inflation) != 4:
            errors.append(
                f"state_zero_inflation needs 4 values, got {len(self.state_zero_inflation)}"
            )

        if not self._validate_matrix(self.session_count_distribution, 4, 3):
            errors.append(
                "session_count_distribution must be 4x3 with rows summing to 1"
            )

        if self.min_gap_seconds > self.max_gap_seconds:
            errors.append("min_gap_seconds must be <= max_gap_seconds")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

    def _validate_matrix(self, matrix: List[List[float]], rows: int, cols: int) -> bool:
        """Check matrix dimensions and row sums."""
        if len(matrix) != rows:
            return False
        for row in matrix:
            if len(row) != cols or not (0.99 <= sum(row) <= 1.01):
                return False
        return True

    @staticmethod
    def deserialize(data: dict) -> "Config":
        """Load Config from dictionary with type conversions."""
        data = data.copy()

        # Convert ISO strings to datetime
        for key in ["start_date", "end_date"]:
            if isinstance(data.get(key), str):
                data[key] = datetime.fromisoformat(data[key])

        # Convert path string
        if isinstance(data.get("repo_path"), str):
            data["repo_path"] = Path(data["repo_path"])

        # Convert breaks
        if "breaks" in data:
            breaks = []
            for item in data["breaks"]:
                if isinstance(item, dict):
                    start = item["start"]
                    end = item["end"]
                    factor = item["factor"]
                else:
                    start, end, factor = item

                if isinstance(start, str):
                    start = datetime.fromisoformat(start)
                if isinstance(end, str):
                    end = datetime.fromisoformat(end)

                breaks.append(Break(start=start, end=end, factor=factor))
            data["breaks"] = breaks

        config = Config(**data)
        config.validate()
        return config

    def serialize(self) -> dict:
        """Save Config to dictionary with type conversions."""
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        data["repo_path"] = str(self.repo_path)
        data["breaks"] = [
            {
                "start": brk.start.isoformat(),
                "end": brk.end.isoformat(),
                "factor": brk.factor,
            }
            for brk in self.breaks
        ]
        return data


class MarkovChain:
    """Models day-to-day activity state transitions."""

    STATES = ["off", "quiet", "normal", "busy"]

    def __init__(
        self, transition_matrix: List[List[float]], seed: Optional[int] = None
    ):
        self.transition_matrix = transition_matrix
        self.rng = random.Random(seed)
        self.current_state = 2  # Start in "normal" state

    def next_state(self) -> int:
        """Transition to next state and return its index."""
        probabilities = self.transition_matrix[self.current_state]
        self.current_state = self.rng.choices(
            range(len(probabilities)), weights=probabilities
        )[0]
        return self.current_state


class Sampler:
    """Utilities for sampling from statistical distributions."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def poisson(self, lam: float) -> int:
        """Sample from Poisson distribution using Knuth algorithm."""
        if lam <= 0:
            return 0

        L = math.exp(-lam)
        k = 0
        p = 1.0

        while p > L:
            p *= self.rng.random()
            k += 1

        return k - 1

    def zero_inflated_negative_binomial(
        self, mean: float, dispersion: float, zero_inflation: float
    ) -> int:
        """Sample from ZINB distribution."""
        # Zero inflation
        if self.rng.random() < zero_inflation:
            return 0

        # Negative binomial via gamma-Poisson mixture
        if mean > 0:
            lam = self.rng.gammavariate(dispersion, mean / dispersion)
            return self.poisson(lam)

        return 0

    def lognormal_bounded(
        self, mu: float, sigma: float, min_val: float, max_val: float
    ) -> float:
        """Sample from log-normal distribution with bounds."""
        value = self.rng.lognormvariate(mu, sigma)
        return max(min_val, min(max_val, value))


class CommitGenerator:
    """Generates realistic commit schedules using Markov chains."""

    def __init__(self, config: Config):
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.sampler = Sampler(self.rng)
        self.markov = MarkovChain(config.transition_matrix, config.random_seed)

    def generate_schedule(self) -> List[Commit]:
        """Generate complete commit schedule for date range."""
        schedule = []
        current_date = self.config.start_date.date()
        end_date = self.config.end_date.date()

        while current_date <= end_date:
            day_commits = self._generate_day_commits(current_date)
            schedule.extend(day_commits)
            current_date += timedelta(days=1)

        return sorted(schedule, key=lambda c: c.timestamp)

    def _get_break_factor(self, date: datetime.date) -> float:
        """Get activity scale factor for a date."""
        for brk in self.config.breaks:
            if brk.start.date() <= date <= brk.end.date():
                return brk.factor
        return 1.0

    def _generate_day_commits(self, date: datetime.date) -> List[Commit]:
        """Generate all commits for a single day."""
        state = self.markov.next_state()

        if state == 0:  # Off state
            return []

        # Calculate expected commits for the day
        weekday_weight = self.config.weekly_weights[date.weekday()]
        break_factor = self._get_break_factor(date)
        mean = self.config.state_means[state] * weekday_weight * break_factor

        # Sample commit count
        count = self.sampler.zero_inflated_negative_binomial(
            mean=mean,
            dispersion=self.config.state_dispersion,
            zero_inflation=self.config.state_zero_inflation[state],
        )

        if count == 0:
            return []

        # Generate timestamps
        timestamps = self._generate_timestamps(date, state, count)

        # Create commit objects
        return [
            Commit(timestamp=ts, message=f"update {self._random_hash()}")
            for ts in timestamps
        ]

    def _generate_timestamps(
        self, date: datetime.date, state: int, count: int
    ) -> List[datetime]:
        """Generate commit timestamps for a day."""
        # Determine number of work sessions
        session_probs = self.config.session_count_distribution[state]
        num_sessions = self.rng.choices([1, 2, 3], weights=session_probs)[0]
        num_sessions = min(num_sessions, count)

        # Distribute commits across sessions
        commits_per_session = self._distribute_across_sessions(count, num_sessions)

        # Generate timestamps for each session
        timestamps = []
        for session_commits in commits_per_session:
            if session_commits > 0:
                session_timestamps = self._generate_session_timestamps(
                    date, session_commits
                )
                timestamps.extend(session_timestamps)

        return sorted(timestamps)

    def _distribute_across_sessions(self, total: int, num_sessions: int) -> List[int]:
        """Randomly distribute commits across sessions."""
        if num_sessions == 1:
            return [total]

        distribution = [0] * num_sessions
        for _ in range(total):
            distribution[self.rng.randint(0, num_sessions - 1)] += 1

        return distribution

    def _generate_session_timestamps(
        self, date: datetime.date, count: int
    ) -> List[datetime]:
        """Generate timestamps for commits within a single session."""
        session_start = self._sample_session_start(date)

        timestamps = []
        current_time = session_start

        for _ in range(count):
            timestamps.append(current_time)

            # Add gap to next commit
            gap_seconds = self._sample_commit_gap()
            current_time += timedelta(seconds=gap_seconds)

        return timestamps

    def _sample_session_start(self, date: datetime.date) -> datetime:
        """Sample a session start time with jitter."""
        # Choose random session window
        hour, minute = self.rng.choice(self.config.session_windows)
        base_minutes = hour * 60 + minute

        # Add Gaussian jitter
        jitter = self.rng.gauss(0, self.config.session_std_minutes)
        total_minutes = max(0, min(23 * 60 + 59, base_minutes + jitter))

        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        seconds = self.rng.randint(0, 59)

        return datetime.combine(date, datetime.min.time()) + timedelta(
            hours=hours, minutes=minutes, seconds=seconds
        )

    def _sample_commit_gap(self) -> int:
        """Sample time gap between commits (log-normal distribution)."""
        mu = math.log(7 * 60)  # 7 minute mean
        sigma = 0.8

        gap = self.sampler.lognormal_bounded(
            mu, sigma, self.config.min_gap_seconds, self.config.max_gap_seconds
        )

        return int(gap)

    def _random_hash(self) -> str:
        """Generate random 6-character hex hash."""
        return "".join(self.rng.choices("0123456789abcdef", k=6))


class GitCommitter:
    """Executes Git commits with backdated timestamps."""

    def __init__(self, config: Config):
        self.config = config

    def execute_schedule(self, schedule: List[Commit]) -> None:
        """Create all commits in the schedule."""
        if self.config.dry_run:
            return

        self._ensure_repo_exists()

        total = len(schedule)
        for i, commit in enumerate(schedule, 1):
            self._create_commit(commit)
            if i % 10 == 0:
                print(f"Created {i}/{total} commits...", end="\r")

        print(f"\nSuccessfully created {total} commits")

    def _ensure_repo_exists(self) -> None:
        """Initialize Git repository if needed."""
        repo_path = self.config.repo_path

        if not (repo_path / ".git").exists():
            print(f"Initializing Git repository at {repo_path}")
            repo_path.mkdir(parents=True, exist_ok=True)
            self._run_git(["git", "init"])
            self._run_git(["git", "checkout", "-b", self.config.branch])

    def wipe_commits(self) -> None:
        """Remove all commits from repository."""
        print(f"Wiping all commits from {self.config.repo_path}...")

        try:
            self._run_git(["git", "checkout", "--orphan", "temp-branch"])
            self._run_git(["git", "rm", "-rf", "."], check=False)
            self._run_git(["git", "branch", "-D", self.config.branch], check=False)
            self._run_git(["git", "branch", "-m", self.config.branch])
            print("All commits wiped successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error wiping commits: {e}")
            raise

    def _create_commit(self, commit: Commit) -> None:
        """Create a single commit with backdated timestamp."""
        env = os.environ.copy()
        timestamp_str = commit.timestamp.strftime("%Y-%m-%dT%H:%M:%S%z")

        env.update(
            {
                "GIT_AUTHOR_NAME": self.config.user_name,
                "GIT_AUTHOR_EMAIL": self.config.user_email,
                "GIT_AUTHOR_DATE": timestamp_str,
                "GIT_COMMITTER_NAME": self.config.user_name,
                "GIT_COMMITTER_EMAIL": self.config.user_email,
                "GIT_COMMITTER_DATE": timestamp_str,
            }
        )

        self._run_git(["git", "commit", "--allow-empty", "-m", commit.message], env=env)

    def _run_git(self, cmd: List[str], env: dict = None, check: bool = True) -> None:
        """Run a Git command in the repository directory."""
        subprocess.run(
            cmd, cwd=self.config.repo_path, env=env, check=check, capture_output=True
        )


class Visualizer:
    """Renders commit schedule as GitHub-style contribution graph."""

    # Color scheme (GitHub-style green)
    COLOR_MIN = (0, 0, 0)
    COLOR_MAX = (57, 211, 83)
    BLOCK = "█"

    def __init__(
        self, schedule: List[Commit], start_date: datetime, end_date: datetime
    ):
        self.schedule = schedule
        self.start_date = start_date.date()
        self.end_date = end_date.date()
        self.commit_map = self._build_commit_map()
        self.min_count = min(self.commit_map.values())
        self.max_count = max(self.commit_map.values())

    def _build_commit_map(self) -> Dict[datetime.date, int]:
        """Create mapping of date to commit count."""
        commit_map = {}

        # Initialize all dates with zero
        current = self.start_date
        while current <= self.end_date:
            commit_map[current] = 0
            current += timedelta(days=1)

        # Count commits per day
        for commit in self.schedule:
            date = commit.timestamp.date()
            commit_map[date] += 1

        return commit_map

    def render(self) -> str:
        """Render the contribution graph adaptively."""
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80

        total_days = (self.end_date - self.start_date).days + 1
        total_weeks = (total_days + 6) // 7
        horizontal_width = 8 + total_weeks * 2

        lines = []

        if horizontal_width <= terminal_width:
            lines += self._render_horizontal()
        elif terminal_width >= 70:
            lines += self._render_quarterly()
        else:
            lines += self._render_vertical()

        lines += self._render_statistics()

        return "\n".join(lines)

    def _render_horizontal(self) -> List[str]:
        """Render full horizontal GitHub-style graph."""
        lines = [self._bold("Commit Graph Preview\n")]

        weeks = self._build_weeks(self.start_date, self.end_date)

        # Month labels
        lines.append(self._muted(self._build_month_labels(weeks)))

        # Day rows
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for day_idx in range(7):
            line = self._muted(day_labels[day_idx]) + " "
            for week in weeks:
                count = week[day_idx]
                line += "  " if count is None else self._colored_block(count) * 2
            lines.append(line)

        lines.append("")
        lines.append(self._render_legend())

        return lines

    def _render_quarterly(self) -> List[str]:
        """Render in quarterly sections."""
        lines = [
            self._bold("Commit Graph Preview") + " " + self._muted("(quarterly view)\n")
        ]

        current_date = self.start_date

        while current_date <= self.end_date:
            year = current_date.year
            quarter = (current_date.month - 1) // 3 + 1
            quarter_end = self._get_quarter_end(year, quarter)
            quarter_end = min(quarter_end, self.end_date)

            lines.append(
                self._bold(f"Q{quarter} {year}")
                + self._muted(
                    f" {current_date.strftime('%b %d')} - {quarter_end.strftime('%b %d, %Y')}"
                )
            )

            weeks = self._build_weeks(current_date, quarter_end)
            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            for day_idx in range(7):
                line = self._muted(day_labels[day_idx]) + " "
                for week in weeks:
                    count = week[day_idx]
                    line += "  " if count is None else self._colored_block(count) * 2
                lines.append(line)

            lines.append("")
            current_date = quarter_end + timedelta(days=1)

        return lines

    def _render_vertical(self) -> List[str]:
        """Render in vertical/transposed layout."""
        lines = [
            self._bold("Commit Graph Preview") + " " + self._muted("(vertical view)\n")
        ]

        lines.append(self._bold("Date") + "   " + self._muted("M T W T F S S"))

        weeks = self._build_weeks(self.start_date, self.end_date)
        current_date = self.start_date

        # Find first Monday
        while current_date.weekday() != 0:
            current_date -= timedelta(days=1)

        for week in weeks:
            date_str = self._muted(current_date.strftime("%b %d"))
            line = f"{date_str} "

            for day_idx in range(7):
                count = week[day_idx]
                line += "  " if count is None else self._colored_block(count) * 2

            lines.append(line)
            current_date += timedelta(days=7)

        lines.append("")

        return lines

    def _render_statistics(self) -> List[str]:
        """Render commit statistics."""
        active_days = sum(1 for count in self.commit_map.values() if count > 0)
        avg_commits = len(self.schedule) / max(active_days, 1)

        return [
            self._bold("Statistics\n"),
            f"  Total commits: {len(self.schedule)}",
            f"  Days with commits: {active_days}",
            f"  Average per active day: {avg_commits:.1f}",
            f"  Max commits in a day: {self.max_count}",
        ]

    def _render_legend(self) -> str:
        """Render color gradient legend."""
        legend = "Less "
        num_blocks = 9

        for i in range(num_blocks):
            count = 0 if i == 0 else int((i / (num_blocks - 1)) * self.max_count)
            legend += self._colored_block(count)

        legend += " More\n"
        return legend

    def _build_weeks(
        self, start: datetime.date, end: datetime.date
    ) -> List[List[Optional[int]]]:
        """Build week data structure for date range."""
        current = start
        while current.weekday() != 0:
            current -= timedelta(days=1)

        weeks = []
        week = [None] * current.weekday()

        while current <= end:
            if current >= start:
                week.append(self.commit_map.get(current, 0))
            else:
                week.append(None)

            if len(week) == 7:
                weeks.append(week)
                week = []

            current += timedelta(days=1)

        if week:
            week.extend([None] * (7 - len(week)))
            weeks.append(week)

        return weeks

    def _build_month_labels(self, weeks: List[List[Optional[int]]]) -> str:
        """Build month label row for horizontal view."""
        months = [(self.start_date.strftime("%b"), 0)]
        current_col = 0

        for date in self.commit_map:
            month_abbr = date.strftime("%b")
            if month_abbr != months[-1][0]:
                if current_col - months[-1][1] < 2:
                    months[-1] = ("", months[-1][1])
                months.append((month_abbr, current_col))
            if date.weekday() == 6:
                current_col += 1

        labels = " " * 4
        for i in range(len(months) - 1):
            labels += months[i][0]
            labels += " " * ((months[i + 1][1] - months[i][1]) * 2 - len(months[i][0]))
        labels += months[-1][0]

        return labels

    def _get_quarter_end(self, year: int, quarter: int) -> datetime.date:
        """Get last day of quarter."""
        end_month = quarter * 3
        if end_month == 12:
            return datetime(year, 12, 31).date()
        return (datetime(year, end_month + 1, 1) - timedelta(days=1)).date()

    def _colored_block(self, count: int) -> str:
        """Get colored block for commit count."""
        ratio = (count - self.min_count) / max(self.max_count - self.min_count, 1)
        ratio = math.sqrt(ratio)  # Non-linear scaling for better visibility

        r = int(self.COLOR_MIN[0] + (self.COLOR_MAX[0] - self.COLOR_MIN[0]) * ratio)
        g = int(self.COLOR_MIN[1] + (self.COLOR_MAX[1] - self.COLOR_MIN[1]) * ratio)
        b = int(self.COLOR_MIN[2] + (self.COLOR_MAX[2] - self.COLOR_MIN[2]) * ratio)

        return f"\033[38;2;{r};{g};{b}m{self.BLOCK}\033[0m"

    def _bold(self, text: str) -> str:
        return f"\033[1m{text}\033[0m"

    def _muted(self, text: str) -> str:
        return f"\033[90m{text}\033[0m"


def create_default_config() -> Config:
    """Create default configuration."""
    return Config(
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        timezone=time.strftime("%Z"),
        weekly_weights=[1.0] * 7,
        breaks=[],
        transition_matrix=[
            [0.85, 0.10, 0.05, 0.00],  # off
            [0.15, 0.60, 0.20, 0.05],  # quiet
            [0.05, 0.15, 0.65, 0.15],  # normal
            [0.00, 0.05, 0.25, 0.70],  # busy
        ],
        state_means=[0, 1, 3, 7],
        state_dispersion=1.2,
        state_zero_inflation=[1.0, 0.3, 0.1, 0.05],
        session_windows=[(10, 30), (15, 0), (20, 30)],
        session_std_minutes=60,
        min_gap_seconds=45,
        max_gap_seconds=40 * 60,
        session_count_distribution=[
            [1.0, 0.0, 0.0],  # off
            [0.8, 0.2, 0.0],  # quiet
            [0.3, 0.5, 0.2],  # normal
            [0.1, 0.4, 0.5],  # busy
        ],
        user_name="",
        user_email="",
        repo_path=Path.cwd() / "commits-repo",
        branch="main",
        random_seed=None,
        dry_run=False,
    )


def parse_date(date_str: str) -> datetime:
    """Parse date from various formats."""
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_str}")


def get_git_config(key: str) -> Optional[str]:
    """Get value from Git config."""
    try:
        result = subprocess.run(
            ["git", "config", key], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no with default."""
    suffix = "([yes]/no)" if default else "(yes/[no])"

    while True:
        response = input(f"{question} {suffix}: ").strip().lower()
        if response == "":
            return default
        if response in ["yes", "y"]:
            return True
        if response in ["no", "n"]:
            return False
        print("Please answer 'yes' or 'no'.")


def prompt_repo_action(repo_path: Path) -> str:
    """Prompt for action when repo exists."""
    print(f"Repository already exists: {repo_path}")

    try:
        result = subprocess.run(
            ["git", "rev-list", "--all", "--count"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            count = int(result.stdout.strip())
            if count > 0:
                print(f"  Current commits: {count}")
    except Exception:
        pass

    print("Actions:")
    print("  wipe   - Delete all existing commits")
    print("  append - Add to existing history")
    print("  cancel - Exit without changes")

    while True:
        response = input("Choose action ([cancel]/wipe/append): ").strip().lower()
        if response in ["", "cancel", "c"]:
            return "cancel"
        if response in ["wipe", "w"]:
            return "wipe"
        if response in ["append", "a"]:
            return "append"
        print("Please choose 'wipe', 'append', or 'cancel'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Commitose: Markov chain-based commit history simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Date arguments
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")

    # Git arguments
    parser.add_argument("--repo-path", type=Path, help="Path to Git repository")
    parser.add_argument("--user-name", help="Git author name")
    parser.add_argument("--user-email", help="Git author email")
    parser.add_argument("--branch", help="Git branch name (default: main)")

    # Generation arguments
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--break",
        action="append",
        dest="breaks",
        help="Activity scaling period (start:end[:factor] in YYYY-MM-DD:YYYY-MM-DD[:FLOAT] format, factor defaults to 0)",
    )

    # Configuration file
    parser.add_argument("--config", type=Path, help="Load configuration from JSON")
    parser.add_argument("--save-config", type=Path, help="Save configuration to JSON")

    # Runtime options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate schedule without making commits",
    )

    args = parser.parse_args()

    # Load or create configuration
    config = load_config(args)

    # Handle Git author info
    config = configure_git_author(config, args, parser)

    # Apply command-line overrides
    config = apply_cli_overrides(config, args)

    # Validate after all modifications
    config.validate()

    # Save config if requested
    if args.save_config:
        with open(args.save_config, "w") as f:
            json.dump(config.serialize(), f, indent=2)
        print(f"Configuration saved to {args.save_config}")

    # Generate schedule
    print("Generating commit schedule...")
    generator = CommitGenerator(config)
    schedule = generator.generate_schedule()

    # Visualize
    visualizer = Visualizer(schedule, config.start_date, config.end_date)
    print("\n" + visualizer.render())

    # Handle existing repository
    if not config.dry_run and (config.repo_path / ".git").exists():
        action = prompt_repo_action(config.repo_path)
        if action == "cancel":
            print("Aborted.")
            return
        elif action == "wipe":
            committer = GitCommitter(config)
            committer.wipe_commits()

    # Confirm before committing
    if not config.dry_run:
        if not prompt_yes_no("\nProceed with creating commits?"):
            print("Aborted.")
            return

    # Execute commits
    committer = GitCommitter(config)
    committer.execute_schedule(schedule)

    if not config.dry_run:
        print(f"\nRepository location: {config.repo_path}")
        print("To push to GitHub:")
        print(f"  cd {config.repo_path}")
        print("  git remote add origin <url>")
        print(f"  git push -u origin {config.branch}")


def load_config(args: argparse.Namespace) -> Config:
    """Load configuration from file or create default."""
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)

        # Merge with defaults for missing fields
        default_dict = create_default_config().serialize()
        default_dict.update(config_dict)

        return Config.deserialize(default_dict)

    return create_default_config()


def configure_git_author(
    config: Config, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Config:
    """Configure Git author name and email."""
    # Apply command-line overrides first
    if args.user_name:
        config.user_name = args.user_name
    if args.user_email:
        config.user_email = args.user_email

    # If still missing, try Git config
    if not config.user_name or not config.user_email:
        git_name = get_git_config("user.name")
        git_email = get_git_config("user.email")

        if git_name and git_email:
            print("Git configuration detected:")
            print(f"  Name:  {git_name}")
            print(f"  Email: {git_email}")

            if prompt_yes_no("Use these values for commits?"):
                if not config.user_name:
                    config.user_name = git_name
                if not config.user_email:
                    config.user_email = git_email
            else:
                # User declined
                missing = []
                if not config.user_name:
                    missing.append("--user-name")
                if not config.user_email:
                    missing.append("--user-email")
                parser.error(f"{' and '.join(missing)} required")
        else:
            # No Git config found
            missing = []
            if not config.user_name:
                missing.append("--user-name (Git config user.name not set)")
            if not config.user_email:
                missing.append("--user-email (Git config user.email not set)")
            parser.error(f"Required: {', '.join(missing)}")

    return config


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply command-line argument overrides to config."""
    if args.start_date:
        config.start_date = parse_date(args.start_date)

    if args.end_date:
        config.end_date = parse_date(args.end_date)

    if args.repo_path:
        config.repo_path = args.repo_path

    if args.branch:
        config.branch = args.branch

    if args.seed is not None:
        config.random_seed = args.seed

    if args.dry_run:
        config.dry_run = True

    # Parse breaks
    if args.breaks:
        config.breaks = []
        for brk in args.breaks:
            parts = brk.split(":")
            if len(parts) == 2:
                start_str, end_str = parts
                factor = 0.0
            elif len(parts) == 3:
                start_str, end_str, factor_str = parts
                factor = float(factor_str)
            else:
                raise ValueError(
                    f"Invalid break format: {brk}. Expected START:END or START:END:FACTOR"
                )

            config.breaks.append(
                Break(
                    start=parse_date(start_str), end=parse_date(end_str), factor=factor
                )
            )

    return config


if __name__ == "__main__":
    main()
