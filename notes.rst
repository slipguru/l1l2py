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


Ranges
------

range = [start, end, step]
linear = range(1):range(3):range(2);
geometric = [range(1)
                range(1) *
                    ( (range(2)/range(1))^(1/(range(3)-1)) ).^(1:(tau_range(3)-1))];

          [ start * [(end/start)^(1/(step-1))]^0,
            start * [(end/start)^(1/(step-1))]^1,
            start * [(end/start)^(1/(step-1))]^2,
            ...
            start * [(end/start)^(1/(step-1))]^(step-1) ]

[2, 10, 20] # 20 numeri tra 2 e 10 in serie geometrica
end/start = 10/2 = 5
ratio= 5^(1/19)
[2*ratio^0 2*ratio^1, 2*ratio^2, ..., 2*ratio^19)
