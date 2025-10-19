# Commitose

Markov chain-based commit history simulator for realistic GitHub contribution graphs.

> [!WARNING]
> This tool was developed for educational and testing purposes only. The author does not encourage or endorse any use that involves dishonesty or false pretenses, including:
> 
> - Misrepresenting your contribution history to employers or collaborators
> - Creating fake activity to game GitHub metrics or streaks
> - Deceiving others about your actual work or skill level
> 
> Please consider the implications of synthetic commit history and be transparent about its nature when relevant.

## Features

- **Markov chain timing:** 4-state model (off, quiet, normal, busy) for realistic activity patterns
- **Session clustering:** Commits grouped into 1-3 work sessions per day with realistic gaps
- **Weekly rhythms:** Configurable weekday vs. weekend activity
- **Vacation periods:** Specify inactive time blocks
- **Adaptive visualization:** Preview commit graph with automatic terminal width handling

## Installation

**Prerequisite:** Python 3+

```bash
git clone https://github.com/dnlzro/commitose.git
cd commitose
chmod +x commitose.py
```

## Quick start

> [!NOTE]
> If you have not configured your Git user name and email, you must provide them using the `--user-name` and `--user-email` options.

```bash
# Preview a year of commits (dry run)
./commitose.py --dry-run

# Generate commits in specific date range
./commitose.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31

# Specify identity explicitly
./commitose.py \
  --user-name "Daniel Lazaro" \
  --user-email "git@dlazaro.ca"

# Add vacation periods
./commitose.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --vacation 2025-07-15:2025-07-29 \
  --vacation 2025-12-20:2026-01-02
```

**See also:** [Advanced configuration](#advanced-configuration)

## Command line options

- `--user-name NAME`: Git user name (will prompt to use Git config if available)
- `--user-email EMAIL`: Git user email (will prompt to use Git config if available)
- `--start-date YYYY-MM-DD`: Start date for commit generation (default: 365 days ago)
- `--end-date YYYY-MM-DD`: End date for commit generation (default: today)
- `--vacation START:END`: Vacation period (repeatable)
- `--seed INT`: Random seed for reproducibility
- `--dry-run`: Preview without creating commits
- `--repo-path PATH`: Repository location (default: `./commits-repo`)
- `--branch NAME`: Branch name (default: `main`)
- `--config FILE`: Load configuration from JSON file
- `--save-config FILE`: Save configuration to JSON file

## Pushing to GitHub

After generating commits:

```bash
cd commits-repo  # or your --repo-path
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

> [!IMPORTANT]
> The email (`--user-email`) must match a **verified email** on your GitHub account for commits to appear in your contribution graph.

## Advanced configuration

For fine-grained control, create a JSON config file specifying one or more of the following options:

```json
{
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "user_name": "Your Name",
  "user_email": "your.email@example.com",
  "repo_path": "./commits-repo",
  "branch": "main",
  "timezone": "UTC",
  "random_seed": 42,
  "weekly_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "vacation_blocks": [],
  "transition_matrix": [
    [0.85, 0.10, 0.05, 0.00],
    [0.15, 0.60, 0.20, 0.05],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.25, 0.70]
  ],
  "state_means": [0, 1, 3, 7],
  "state_dispersion": 1.2,
  "state_zero_inflation": [1.0, 0.3, 0.1, 0.05],
  "session_count_distribution": [
    [1.0, 0.0, 0.0],
    [0.8, 0.2, 0.0],
    [0.3, 0.5, 0.2],
    [0.1, 0.4, 0.5]
  ],
  "session_windows": [[10, 30], [15, 0], [20, 30]],
  "session_std_minutes": 60,
  "min_gap_seconds": 45,
  "max_gap_seconds": 2400
}
```

### Explanation

#### Basic settings

- `start_date`, `end_date`: Date range for commit generation (format: `YYYY-MM-DD`)
- `user_name`, `user_email`: Git author identity
  - **Note:** Email must be verified on GitHub for commits to appear in the contribution graph
- `repo_path`: Local repository path (default: `./commits-repo`)
- `branch`: Git branch name (default: `main`)
- `dry_run`: Preview mode—generates schedule without creating commits (default: `false`)
- `timezone`: Timezone for timestamps (default: system timezone)
- `random_seed`: Seed for reproducibility (default: `null`, i.e., random)

#### Activity patterns

- `weekly_weights`: Relative activity levels for each day (`[Mon, Tue, Wed, Thu, Fri, Sat, Sun]`)
  - Values are multiplied against state means (see below)
- `vacation_blocks`: List of inactive date ranges
  - e.g., `[["2025-07-15", "2025-07-29"], ...]`

#### Markov chain (daily state)

The generator uses a 4-state Markov chain (off, quiet, normal, busy) to model realistic activity patterns:

- `transition_matrix`: State transition probabilities
  - Each row represents `[P(off), P(quiet), P(normal), P(busy)]` from the current state
  - Each row must sum to 1.0
- `state_means`: Expected commits per day for each state `[off, quiet, normal, busy]` (before weekly weighting)
- `state_dispersion`: Controls variance in commit count distribution (lower = more uniform)
- `state_zero_inflation`: Probability of forcing zero commits per state
  - Separates "true off days" from naturally quiet days.

#### Within-day timing

- `session_count_distribution`: Probability of having 1, 2, or 3 work sessions per state
  - Each row `[P(1), P(2), P(3)]` must sum to 1.0
- `session_windows`: Typical session start times, as `[hour, minute]` pairs in 24-hour format (e.g., `[13, 30]` = 1:30 PM)
- `session_std_minutes`: Standard deviation (in minutes) for jitter around `session_windows` times
- `min_gap_seconds`, `max_gap_seconds`: Range for inter-commit gaps within a session (sampled from log-normal distribution)
