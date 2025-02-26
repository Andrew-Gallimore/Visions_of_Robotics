# TODO Items

This is a list of times which need to get done, they may be dynamic, so if you choose to do one, edit this file to have your name next to it, so we know someone is handling that.

Just make sure to stage and then push this change so we can all get it and see that you are doing it.

### List:

- [x] Example todo item

- [ ] Implement actual image scalar function/step, rather than just sampling every 3 points in the searching function if the image scale is 3... bad Andrew. Really, there should be a function called from in runRound() for this, I think.

- [ ] Try different method for making the depth map. The current structure of puttting the points in a grid to then sample isn't perfect, especially when we have either too many points or too few in a grid space. Takes too much fiddling, and it should just work!

- [ ] Check depth is working by just supplying cherry-picked points to see what it thinks the other point is, I think its not actually finding the right point... (as we see by many of the found points all having (0,0) coordinates, not good)

	- [ ] Substep: Add visualizer for a subset of points and their matching points, overlayed over one of the images (likely left). It would be good to get some visualizer put in for inermediary checks to make sure things are working.

- [ ] Secondly, figure out a solid method for transforming the depth outputs into something between 0-255 (and maybe able to do color?). There should be a way to specify a min-distance and a max-distance (in millimeters) it should put into the image.
    - [ ] Figure out the units of the distance outputs we are getting (the values we are putting into our depth map currently). Are they millimeters?