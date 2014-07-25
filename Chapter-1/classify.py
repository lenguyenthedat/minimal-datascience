import pandas as pd
import time

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Read dataset
df = pd.read_csv('./Dataset/Starcraft/SkillCraft1_Dataset_clean.csv')

# Train & Test
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train_df, test_df = df[df['is_train']==True], df[df['is_train']==False]

# Define classifiers
classifiers = [ 
    ExtraTreesClassifier(n_estimators=10),
    RandomForestClassifier(n_estimators=10),
    KNeighborsClassifier(100),
    LDA(),
    QDA(),
    GaussianNB(),
    DecisionTreeClassifier()
]

# Train classifiers
features = ["APM","Age","TotalHours","UniqueHotkeys", "SelectByHotkeys", "AssignToHotkeys",
            "WorkersMade","ComplexAbilitiesUsed","MinimapAttacks","MinimapRightClicks" ]

for classifier in classifiers:
    print classifier.__class__.__name__
    start = time.time()
    classifier.fit(train_df[list(features)], train_df.LeagueIndex)
    print "  -> Training time:", time.time() - start

# Evaluation:
# RMSD / RMSE
for classifier in classifiers:
    print classifier.__class__.__name__
    print np.sqrt(mean(pow(test_df.LeagueIndex - classifier.predict(test_df[features]),2)))

# Crosstab / Contingency Table
for classifier in classifiers:    
    print classifier.__class__.__name__
    print pd.crosstab(test_df.LeagueIndex, classifier.predict(test_df[features]), rownames=["Pred"], colnames=["Actual"])
    pd.crosstab(test_df.LeagueIndex, classifier.predict(test_df[features]), rownames=["Pred"], colnames=["Actual"]).plot()
