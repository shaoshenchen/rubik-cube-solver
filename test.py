import numpy as np

def hashmap(color):
  return {
    'o': "\033[7;31;43m橙\033[0m",
    'w': "\033[1;30;47m白\033[0m",
    'p': "\033[1;10;45m粉\033[0m",
    'y': "\033[1;30;43m黄\033[0m",
    'g': "\033[1;40;42m绿\033[0m",
    'b': "\033[1;40;44m蓝\033[0m"
  }[color]

final_prediction_seq = ['o', 'o', 'o', 'w', 'w', 'w', 'p', 'y' ,'g']
color_list = list(map(hashmap, final_prediction_seq))
for i in range(9):
  print(color_list[i], end='  ')
  if (i+1) % 3 == 0:
    print('\n')