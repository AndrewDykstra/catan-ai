# Changelog

 [1.1.2] - 2024-08-14
### Added
- Created a new directory for data from 10 episodes of training
- Conducted 60 episodes of training and move them to a new directory as well
- Conducted 150 episodes of training and moved to new directory

### Changed
- Corrected the turn order to be a snake turn order 1 -> 2 -> 3 -> 4 -> 4 -> 3 -> 2 -> 1 repeat

### Fixed
- Corrected Changelog for version 1.1.1

## [1.1.1] - 2024-08-14
### Added
- Added multiple forms of tracking metrics on Agent's performance via Matplotlib which export images to results directory
- Step max out at 10,000 steps to prevent infinite loops in training if reached

### Changed
- Modified the turn penalty to -3 instead of -1
- Agents now all use Q-learning and without rulesets in training
- Updatd README milestones accordingly

### Fixed
- N/A

## [1.1.0] - 2024-08-14
### Added
- Agent can now upgrade settlements to cities

### Changed
- Victory point win threshold is now 5

### Fixed
- Agent is now successfully learning!
- Victory point tracking is also corrected

## [1.0.12] - 2024-08-14
### Added
- Debugging print lines to figure out the issue of the games not completing

### Changed
- N/A

### Fixed
- Version number for 1.0.11 push

## [1.0.11] - 2024-08-14
### Added
- N/A

### Changed
- Modified the map, now every player has access to every resource off the starting settlements
- Every player starts with two settlements and two roads now

### Fixed
- N/A

## [1.0.10] - 2024-08-13
### Added
- N/A

### Changed
- Modified the victory point distribution (Still debugging the issue of victory points resetting after 5 have been given out in total)

### Fixed
- N/A

## [1.0.9] - 2024-08-12
### Added
- N/A

### Changed
- Modified the rewards for actions to better train model
- Player starting positions are no longer clumped

### Fixed
- N/A

## [1.0.8] - 2024-08-12
### Added
- Added manual definitions of neighbors for every tile object

### Changed
- Reworked how the map is made entirely, now tiles are individual objects

### Fixed
- Players now gain resources upon each dice roll if they have settlements adjacent to the according tiles

## [1.0.7] - 2024-08-12
### Added
- Logging around dice rolls to ensure players are getting resources
- (Currently having an error where users aren't getting resources from dice rolls)

### Changed
- N/A

### Fixed
- N/A

## [1.0.6] - 2024-08-12
### Added
- Rules around building a road and settlement in valid locations

### Changed
- N/A

### Fixed
- Formatting of ChangeLog

## [1.0.5] - 2024-08-11
### Added
- Trading with the Bank (More actions added)
- Agent with Q-Learning implemented
- Implemented set starting locations
- Adjacency logic
- BUG: Need to debug changing turns between players

### Changed
- N/A

### Fixed
- Formatting of ReadME

## [1.0.4] - 2024-08-01
### Added
- N/A

### Changed
- N/A

### Fixed
- Formatting of ReadME

## [1.0.3] - 2024-08-01
### Added
- Adding to README: feedback from check-in meeting + discussion

### Changed
- N/A

### Fixed
- N/A

## [1.0.2] - 2024-08-01
### Added
- N/A

### Changed
- Updating README

### Fixed
- N/A

## [1.0.1] - 2024-08-01
### Added
- N/A

### Changed
- Formatting of README file

### Fixed
- N/A

## [1.0.0] - 2024-08-01
### Added
- Initial implementation of the game environment.
- Two actions: building roads and settlements.

### Changed
- N/A

### Fixed
- N/A
