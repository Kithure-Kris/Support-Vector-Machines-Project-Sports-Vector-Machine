import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

#Create the labels
#print(aaron_judge.columns)
print(aaron_judge.description.unique())
print(aaron_judge.type.unique())
aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B': 0})
#print(aaron_judge.type)

#Plotting the pitches
print(aaron_judge['plate_x'])
#Drop rows with nulls
aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z', 'type'])
fig, ax = plt.subplots()
plt.scatter(x=aaron_judge.plate_x, y=aaron_judge.plate_z, c= aaron_judge.type, cmap=plt.cm.coolwarm, alpha=0.25)

#Building the SVM
training_set, validation_set = train_test_split(aaron_judge, random_state=1)

classifier = SVC(kernel='rbf', gamma=3, C=1)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
draw_boundary(ax, classifier)
plt.show()

#Optimizing the SVM
print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))
#Accuracy = 0.8355(No gamma or C set)
#With gamma = 100, C=100, Accuracy is 0.7934
#With gamma = 3, C = 1, Accuracy is 0.8341