Implementation Notes
=====================

Standardization & normalization
--------------------------------
Mlpy:   standardize -> unit variance on columns
        normalize -> unit variance on rows
Matlab: normalize -> unit variance on columns (in doc: unit norm!!!)
I:      normalize -> unit norm (length)
        standardize -> unit variance

Questions
---------
+----------+---------+--------------------------------------------------------+
| FUNCTION | ROWS    | DESCRIPTION                                            |
+==========+=========+========================================================+
| KVC_grid | 18, 19  | remove labels with value 0 ?                           |
+----------+---------+--------------------------------------------------------+
| KVC_grid | 23      | no norm=1 normalization ?                              |
+----------+---------+--------------------------------------------------------+
| KVC_grid | 30      | labels not normalized why assumed -1 or 1 ?            |
+----------+---------+--------------------------------------------------------+
| KVC_grid | 25,31,33| range computation!!! ???                               |
+----------+---------+--------------------------------------------------------+


