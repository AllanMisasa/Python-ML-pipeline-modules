import xlsxwriter
from image_fetch import get_url_paths, url_strip 

url = "https://github.com/AllanMisasa/Python-ML-pipeline-modules/blob/main/Computer%20Vision/DD_dataset/transformed"

ext = '.jpg'
image_paths = [url_strip(url) for url in get_url_paths(url, ext)] # Gets the paths of the images from the url
file_names = [s.split('/')[-1] for s in image_paths] # Gets the file names of the images

workbook = xlsxwriter.Workbook('sea_samples_file_names_targets.xlsx')
worksheet = workbook.add_worksheet()
 
# Start from the first cell.
# Rows and columns are zero indexed.
row = 0
column = 0
 
# iterating through content list
for item in file_names:
 
    # write operation perform
    worksheet.write(row, column, item)
 
    # incrementing the value of row by one
    # with each iterations.
    row += 1
     
workbook.close()