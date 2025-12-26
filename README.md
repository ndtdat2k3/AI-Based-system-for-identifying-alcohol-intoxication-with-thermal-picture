# AI-Based-system-for-identifying-alcohol-intoxication-with-thermal-picture
Since the dataset is too big to push to GitHub, I will give you the link to it: https://universe.roboflow.com/drunk-thermal-image/drunk-thermal-image/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
Next, you should:
- Augmentation: Increase the dataset using the Roboflow tool
- Rename datasets and name them Fig_1, Fig_2,...
- Using LabelImg.py tool to draw bounding boxes and save as .xml files.
- Create an annotation.csv file containing all data from those .xml files (use AI to write a Python code to help you with it)
- Upload the dataset to gg drive
- Run the Yolo code on gg colab and download the best.pt file
- Run the Resnet code on gg colab and download the .pth file
- Put the best.pt, .pth and web-app.py file in the same folder and run the web-app file (remember to download the streamlit library)
- That's it, good luck!
