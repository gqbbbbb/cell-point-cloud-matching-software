import pandas as pd
import re

 
def is_float(s):
    pattern = r'^[-+]?[0-9]*\.[0-9]+$'
    return bool(re.match(pattern, s))

def is_integer(s):
    return s.isdigit()

def read_marker(file):
    """
    read '*.marker' file and return pd.DataFrame object
    """
    result = None
    with open(file,'r') as f:
        result_dict = {}
        headers = []
        row_count = 0
        for line in f.readlines():
            if line[:2] == '##':
                if not result_dict:
                    headers = [header.strip() for header in line[2:].strip().split(',')]
                    row_count = len(headers)
                    for header in headers:
                        result_dict[header] = []
                else:
                    raise Exception('duplicated headers in marker file!')
                    return
            else:
                if result_dict:
                    cells = [cell.strip() for cell in line.strip().split(',')]
                    if not len(cells) == row_count:
                        continue
                    for i in range(row_count):
                        header = headers[i]
                        cell = cells[i]
                        if is_integer(cell):
                            cell = int(cell)
                        elif is_float(cell):
                            cell = float(cell)
                        result_dict[header].append(cell)
                else:
                    raise Exception('missing headers before data!')
                    return
        result = pd.DataFrame(result_dict)
    return result

def filter_marker(data):
    data = data.loc[:,['name','x','y','z']]
    data.columns = ['Id','X','Y','Z']
    return data

def save_marker(data,path):
    ##x,y,z,radius,shape,name,comment, color_r,color_g,color_b
    with open(path,'w') as f:
        f.write('##x,y,z,radius,shape,name,comment, color_r,color_g,color_b\n')
        for i in range(len(data)):
            if 'X' in data.columns:
                cells = [data.iloc[i]['X'],data.iloc[i]['Y'],data.iloc[i]['Z'],1,1,
                         data.iloc[i]['Id'],'',255,0,0]
                cells = [str(x) for x in cells]
                f.write(','.join(cells) + '\n')
            elif 'x' in data.columns:
                cells = [data.iloc[i]['x'], data.iloc[i]['y'], data.iloc[i]['z'], 1, 1,
                         data.iloc[i]['name'], '', 255, 0, 0]
                cells = [str(x) for x in cells]
                f.write(','.join(cells) + '\n')


if __name__ == '__main__':
    data = read_marker('inputdata/AtlasVolume-04232024-1014-1582.v3draw.marker')
    data = filter_marker(data)
    save_marker(data,'outputdata/AtlasVolume-04232024-1014-1582.v3draw.marker')
