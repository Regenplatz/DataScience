## Visualize Nelson Trending Rules and

The [Nelson Rules](https://en.wikipedia.org/wiki/Nelson_rules) were coded in [TrendingRules.ipynb](TrendingRules.ipynb) for demonstration purpose. The rules were furthermore coded as a class in [TrendingRules.py](TrendingRules.py). With this, the data can be visualized in a  [line plot](lineplot.py), where red color highlights the presence of the applied Nelson Rule.

For this visualization, [dash](https://plotly.com/dash/) was used. A drop down menu allows for selecting the Nelson Rule of interest.

In addition, also [bar plots](barplot.py) were coded in dash. Bar plots exceeding the lower threshold are colored in yellow while bar plots exceeding the higher threshold are colored in red. No Nelson Rules are applied on the bar plots. However, this kind of plot allows for highlighting data exceeding certain [thresholds](barPlotRules.py) and is therefore additionally included in this application.
