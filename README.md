### Formula 1 Race Winner Prediction System Documentation

## Overview


Purpose: This system uses machine learning to predict the probability of each driver winning an upcoming Formula 1 (F1) race based on historical data. It analyzes factors like past performance, starting grid positions, team strength, and track characteristics to generate win probabilities for drivers in a given race grid.

Key Goal: Help F1 enthusiasts, analysts, or bettors make data-driven predictions. For example, in a real-world scenario like the 2025 Australian Grand Prix, the model might predict Lando Norris has a 25% win chance from pole position, while Max Verstappen has 20% despite stronger recent form, due to track-specific factors.

Technology Stack:


- Language: Python 3.x

- Libraries: pandas, numpy, scikit-learn (for ML pipeline), joblib (for model saving), IPython (for display)

- Model: Gradient Boosting Classifier (from sklearn.ensemble)

- Environment: Designed for Jupyter Notebook or similar (with cell-based execution)

Author/Version: Developed by Adithya (Graduate Student in Information Technology and Management). Version 1.0 (as of September 2025).

System Architecture


The system follows a standard machine learning pipeline: Data Loading → Cleaning → Feature Engineering → Model Training → Prediction. It processes historical F1 data to train a model, then applies it to future race scenarios.

High-Level Flow:


1. Load and Clean Data: Import CSV files with race and qualifying results.

2. Engineer Features: Calculate metrics like rolling averages (e.g., a driver's average points in the last 5 races).

3. Train Model: Use historical data to fit a Gradient Boosting Classifier, which learns patterns of winners.

4. Predict: For a new race grid, prepare data, run predictions, and output win probabilities as a table/CSV.

Real-World Analogy: Imagine a sports coach reviewing game tapes (historical data), calculating player stats (feature engineering), and building a playbook (model) to predict match outcomes. Here, the "playbook" is the ML model predicting F1 winners.

Data Sources

- Input Files:
	- ../data/processed/updated_races.csv: Contains race results (e.g., positions, points, grids) up to the latest season.

	- ../data/processed/updated_qualifying.csv: Qualifying data for driver and team mappings.


- Data Handling:
	- Encodings tried: UTF-8, Latin-1, CP1252, UTF-8-SIG (to handle special characters like accented names).

	- Cleaning: Converts numeric fields, standardizes names (e.g., "Hülkenberg" to "Hulkenberg"), handles missing values.


- Assumptions: Data is pre-processed and includes columns like driverId, constructorId, grid, position, points, season, round, date, circuitId.

Real-World Example: If historical data shows Verstappen winning 60% of races from grid position 1 at Monaco, the model learns this pattern for similar future grids.

Key Features and Engineering


Features are derived from historical data to capture driver form, team performance, and race context.

Core Features:


Feature Name	Description	Type	Real-World Example
grid	Starting position on the grid	Numerical	Norris starts at 1 (pole) → Higher win chance due to clean air at the start.
circuitId	Unique track identifier	Categorical	"monaco" favors qualifying skill over raw speed, like navigating tight city streets.
driverId	Unique driver ID	Categorical	Verstappen's ID encodes his consistent winning history.
constructorId	Unique team ID	Categorical	Red Bull's ID reflects strong car performance, like a top-tier engine in a rally car.
avg_points_last_5	Average points in last 5 races	Numerical (Rolling)	If Hamilton averaged 18 points/race recently, he's in "hot form" like a basketball player on a scoring streak.
avg_position_last_5	Average finish position in last 5 races	Numerical (Rolling)	Average of 3rd place → Consistent podiums, reducing risk of DNFs (Did Not Finish).
avg_grid_last_5	Average starting grid in last 5 races	Numerical (Rolling)	Average grid 2 → Strong qualifier, like a sprinter with a good starting block.
points_standings_prev_race	Championship points before this race	Numerical (Cumulative)	150 points → Title contender with motivation, like a team leading the league standings.
Feature Engineering Details:


- Rolling Windows: Uses pandas' rolling() to compute averages over the last 5 races per driver (shifted by 1 to avoid data leakage).

- Cumulative Sums: Tracks season points up to the previous race.

- Handling Unknowns: Maps new drivers/teams to "unknown_" IDs; rebrands teams (e.g., "Red Bull Racing Honda RBPT" to "red_bull") for consistency.

- Preprocessing: Numerical features imputed with medians; categorical features one-hot encoded and imputed with most frequent values.

Model Details

- Algorithm: Gradient Boosting Classifier (sklearn's GradientBoostingClassifier).
	- Why Gradient Boosting?: It's like building a team of weak predictors that learn from each other's mistakes—excellent for handling mixed data types and non-linear patterns in F1 (e.g., how grid position interacts with track type). It's robust to outliers like crashes or penalties.

	- Hyperparameters: n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42.


- Target Variable: is_winner (binary: 1 if position==1, else 0).

- Training Process: Fits on historical features; saves model as joblogs/f1_winner_predictor_model_gbc_streamlined.joblib.

- Prediction Output: Normalized win probabilities (sum to 100%) for each driver, sorted descending. Merges duplicates by driver name.

- Evaluation: Not explicitly in code (add metrics like accuracy or ROC-AUC for production).

Real-World Analogy: The model is like an F1 simulator game that runs thousands of virtual races based on stats, outputting "Verstappen has a 22% chance to win from grid 3 at Suzuka."

Usage Instructions

1. 


Setup:


	- Install dependencies: pip install pandas numpy scikit-learn joblib ipython.

	- Ensure data files are in ../data/processed/.


2. 
Running the Code (in Jupyter Notebook):


	- Execute cells sequentially:
		- Cells 1-2: Load and clean data.

		- Cell 3: Engineer features and create maps.

		- Cell 4: Train and save model.

		- Cell 5: Define prediction functions.

		- Cell 6: Configure races (e.g., add grids for 2025 races).

		- Cell 7: Run predict_all_races(RACE_CONFIGS) or predict_specific_races(['australian_gp', 'monaco_gp']).


	- Outputs: Markdown tables of predictions + CSV files in predictions_GBC_Streamlined/.


3. 
Custom Predictions:


	- Edit RACE_CONFIGS in Cell 6 to add new races (e.g., specify season, round, circuit_id, description, and grid list).

	- Example Grid Entry: {'driver': 'Lando Norris', 'team': 'McLaren Mercedes', 'grid': 1}.


4. 
Output Example (for 2025 Australian GP):



	| Driver             | Grid | Team                           | Probability |
	|--------------------|------|--------------------------------|-------------|
	| Lando Norris       | 1    | mclaren                        |     25.00%  |
	| Max Verstappen     | 3    | red_bull                       |     20.00%  |
	...



Limitations

- Data Dependency: Relies on up-to-date CSVs; predictions degrade without recent seasons.

- Unforeseen Events: Doesn't account for real-time factors like weather, crashes, or mechanical failures (e.g., a sudden rainstorm at Silverstone).

- New Drivers/Teams: "Unknown" mappings may reduce accuracy for rookies like Kimi Antonelli.

- Overfitting Risk: Model trained on historical data may not adapt to major rule changes (e.g., new car regulations).

- No Multi-Output: Predicts only race winner, not full podium or points.

Potential Improvements

- Add Features: Include weather, tire data, or pit stop strategies (e.g., integrate external APIs).

- Model Enhancements: Experiment with XGBoost for faster training or neural networks for complex interactions.

- Evaluation: Add cross-validation and metrics (e.g., log loss for probabilities).

- SQL Integration: Port feature engineering to SQL queries for database scalability (e.g., use window functions for rolling averages).

- Visualization: Add charts (e.g., via matplotlib) to show probability distributions.

- Real-World Extension: Adapt for other sports, like predicting NBA game winners using player stats.

Contact/Support


For questions, updates, or contributions, contact Adithya (adithya@example.com). This documentation is based on the provided code as of 9/10/2025.


---
This doc is concise yet detailed, with real-world examples to aid explanations. If you need expansions (e.g., code snippets, diagrams, or a PDF version), let me know!