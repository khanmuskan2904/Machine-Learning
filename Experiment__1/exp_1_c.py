# -*- coding: utf-8 -*-
"""exp 1 c

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1m9UacphgXjukv42XFfeE7l8EjEVQV_PQ
"""

def sort_students_by_score(students):
   # Sort the list of tuples using sorted()
   sorted_students = sorted(students, key=lambda x: (-x[1], x[0]))
   return sorted_students
   # Example usage:
students = [
("Neeraj", 98),
("Virat", 90),
("Hardik", 88),
("Bhumrah", 93),
("Sky", 75)
]
sorted_students = sort_students_by_score(students)
print("Sorted students:", sorted_students)
print("UIN : 211P067"