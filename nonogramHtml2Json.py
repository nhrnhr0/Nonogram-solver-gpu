from bs4 import BeautifulSoup
import re
import json
import sys


def html_to_json(file_name):
    f = open(file_name)
    content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    sz_names = soup.find_all('img', {"name": "sz"})
    height = int(re.findall(r'\d+', sz_names[0].get('src'))[-1])
    width = int(re.findall(r'\d+', sz_names[1].get('src'))[-1])

    col_rules = []
    curr_rule = []

    # read col rules:
    col_rules_html = soup.find('div', {"id": "tclue"})
    for html_element in col_rules_html:
        if html_element.get('name') == "cn":
            first, sec = re.findall(r"(\d+)\.(\d+)", html_element.get('id'))[0]
            first, sec = int(first), int(sec)
            if first >= len(col_rules):
                col_rules.append(curr_rule.copy())
                curr_rule.clear()

            curr_rule.append(int(re.findall(r"(\d+)", html_element.get('title'))[0]))
    col_rules.append(curr_rule.copy())
    col_rules = col_rules[1:]



    pass

    row_rules = []

    # read row rules:
    row_rules_html = soup.find('div', {"id": "sclue"})
    for html_element in row_rules_html:
        if html_element.get('name') == "cn":
            first, sec = re.findall(r"(\d+)\.(\d+)", html_element.get('id'))[0]
            first, sec = int(first), int(sec)
            if first >= len(row_rules):
                row_rules.append(curr_rule.copy())
                curr_rule.clear()

            curr_rule.append(int(re.findall(r"(\d+)", html_element.get('title'))[0]))
    row_rules.append(curr_rule.copy())
    row_rules = row_rules[1:]

    my_json = {
        "height": height,
        "width": width,
        "row": row_rules,
        "col": col_rules
    }

    #with open('data.json', 'w') as outfile:
        #json.dump(my_json, outfile)
    json.dump(my_json,sys.stdout)


if __name__ == '__main__':
    html_to_json('data.html')
