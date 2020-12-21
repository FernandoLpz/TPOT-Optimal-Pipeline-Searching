import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

# Methods suggested by tpot in file: "optimal_pipeline.py"
# when exporting the optimal pipeline: model.export("optimal_pipeline.py")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import ZeroCount

class AutoML:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def pipeline_suggested_by_tpot(self):
        # Copied from optimal pipeline suggested by tpot in file "optimal_pipeline.py"
        # Initialize 
        exported_pipeline = make_pipeline(
                        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
                        VarianceThreshold(threshold=0.2),
                        ZeroCount(),
                        GradientBoostingClassifier(learning_rate=1.0, max_depth=10, max_features=0.9000000000000001, min_samples_leaf=16, min_samples_split=3, n_estimators=100, subsample=0.7000000000000001)
                        )
        # Init training
        exported_pipeline.fit(self.x_train, self.y_train)
        
        print(f"Train acc: {exported_pipeline.score(self.x_train, self.y_train)}")
        print(f"Test acc: {exported_pipeline.score(self.x_test, self.y_test)}")
    
    def pipeline_optimization(self):
        # Initialize the model
        self.model = TPOTClassifier(generations=5, 
                            population_size=50, 
                            cv=5, 
                            scoring='accuracy', 
                            verbosity=2, 
                            random_state=1, 
                            n_jobs=-1)
        
        # Starts optimization
        self.model.fit(self.x_train, self.y_train)

    
    def load_data(self):
        # Source: https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame

        df = pd.read_csv('data/tic-tac-toe.csv', index_col=False)
        df['Class'] = df['Class'].replace(['negative', 'positive'], [0, 1])
        df = pd.get_dummies(df, columns=['top-left-square',
                                        'top-middle-square',
                                        'top-right-square',
                                        'middle-left-square',
                                        'middle-right-square',
                                        'bottom-left-square',
                                        'bottom-middle-square',
                                        'bottom-right-square'])
        x = df.drop(['Class'], axis=1).values
        y = df['Class'].values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)

if __name__ == "__main__":
    automl = AutoML()
    automl.load_data()
    automl.pipeline_optimization()
    # automl.train_suggested_tpot()