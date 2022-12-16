# metaData.txt 정체코드 (re)
import re

f = open("data/metaList3.txt", "r")
w = open("data/metaList4.txt", "w")

lines = f.readlines()
for line in lines:
    res = re.sub("'|,| ", "", line)

    w.write(res)

f.close()
w.close()