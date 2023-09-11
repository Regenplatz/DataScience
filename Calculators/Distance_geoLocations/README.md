## Calculate Linear Distance Between Two Geolocations

You would like to know how far two locations are geographically dispersed from each other?

You have both geocoordinates at hand? Then insert them as *coord1* and *coord2* in the main function of [LinearDistance.py](LinearDistance.py) and run the code. The result is printed on the console.

**What is the code doing?** <br>
First, the code translates the `coordinates` to `degrees, minutes and seconds` which are further converted to `decimals`. These are then translated to `radians` to finally calculate the linear distance. For this, also *earth curvature* is taken into consideration
