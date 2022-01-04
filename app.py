from flask import Flask,render_template,request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/")
def home():
    
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    import pandas as pd
    if request.method=='POST':
        result=request.form
        if(result['classifier'] != "" and result['dataset_path'] != ""):
            f = request.files['dataset']
            f.save(secure_filename(f.filename))
            dataset = pd.read_csv(f.filename)
            target_pos = len(dataset.columns)-1
            y = dataset.iloc[:,target_pos]
            x = dataset.drop(dataset.columns[target_pos], axis=1)
            print(y)
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

            prediction = []
            score = 0.0
            tree_image = False
            if(result['classifier'] == "knn"):
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(x_train, y_train)
                import pickle
                pickle.dump(knn, open('knn_model.pkl','wb'))
                knn_model = pickle.load(open('knn_model.pkl','rb'))

                prediction = knn_model.predict(x_test)
                score = knn_model.score(x_test,y_test)
            elif(result['classifier'] == "naive-bayes"):
                from sklearn.naive_bayes import GaussianNB
                gnb = GaussianNB()
                gnb.fit(x_train, y_train)
                import pickle
                pickle.dump(gnb, open('gnb_model.pkl','wb'))
                gnb_model = pickle.load(open('gnb_model.pkl','rb'))

                prediction = gnb_model.predict(x_test)
                score = gnb_model.score(x_test,y_test)
            elif(result['classifier'] == "decision-tree"):
                
                from sklearn.tree import  DecisionTreeClassifier, plot_tree
                clf = DecisionTreeClassifier()
                clf = clf.fit(x_train, y_train)
                
                import pickle
                pickle.dump(clf, open('clf_model.pkl','wb'))
                clf_model = pickle.load(open('clf_model.pkl','rb'))

                prediction = clf_model.predict(x_test)
                score = clf_model.score(x_test,y_test)

                from sklearn import tree
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(25,20))
                _ = tree.plot_tree(clf_model, filled=True)
                fig.savefig("static/decision_tree.jpg")
                tree_image = True
            
            from sklearn.metrics import confusion_matrix
            _confusion_matrix = confusion_matrix(y_test, prediction)

            from sklearn.metrics import classification_report
            _classification_report = classification_report(y_test, prediction, output_dict=True)

            import pandas 
            _classification_report_dataframe = pandas.DataFrame(_classification_report).transpose()
            _classification_report_dataframe = _classification_report_dataframe.sort_values(by=['f1-score'], ascending=False)

            _classification_report_html = _classification_report_dataframe.to_html()
                
            return render_template('home.html', dataset = result['dataset_path'], classifier = result['classifier'], classification_report =_classification_report_html, accuracy = score, confusion_matrix = _confusion_matrix, decision_tree = tree_image)
        
            # return "uploded"
        else:
            return render_template('home.html')




if __name__ == "__main__":
    app.run(debug = True, port = 8000)