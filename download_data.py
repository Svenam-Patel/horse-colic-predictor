"""
download_data.py
Downloads the real, public UCI "Horse Colic" dataset (300 horses).
Source: UCI Machine Learning Repository (Mary McLeish & Matt Cecile,
University of Guelph).  File is mirrored on GitHub; the original is at
https://archive.ics.uci.edu/dataset/47/horse+colic

If this download fails, get horse-colic.data from the UCI page and save it
as data/horse-colic.csv (comma separated, no header, '?' = missing).
"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv"
os.makedirs("data", exist_ok=True)
urllib.request.urlretrieve(URL, "data/horse-colic.csv")
print("Saved data/horse-colic.csv")
