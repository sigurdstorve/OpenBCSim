# == Principle ==
# 1. Generate a mathematical description of the boundary of a closed solid in space.
# 2. Fill the bounding box of the mathematical model with uniformly
#    distributed random points.
# 3. Use the model to filter away the points not within the model.
# 4. Apply geometric transformations to the remaining scatterers to
#    simulate motion.  

class Ellipsoid:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        
        # Compute coeffs needed in parameric formula
        self.a = 0.5*(self.x_max - self.x_min)
        self.x0 = self.x_min + self.a
        self.b = 0.5*(self.y_max - self.y_min)
        self.y0 = self.y_min + self.b 
        self.c = 0.5*(self.z_max - self.z_min)
        self.z0 = self.z_min + self.c

    def is_inside(self, x, y, z):
        temp = ((x-self.x0)/self.a)**2 + ((y-self.y0)/self.b)**2 + ((z-self.z0)/self.c)**2
        return temp <= 1.0

class CappedZEllipsoid:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, z_ratio):
        assert(z_ratio >= 0.0 and z_ratio <= 1.0)
        self.ellipsoid = Ellipsoid(x_min, x_max, y_min, y_max, z_min, z_max)
        z_len = z_max - z_min
        self.z_cap = z_min + z_len*z_ratio
    
    def is_inside(self, x, y, z):
        return self.ellipsoid.is_inside(x, y, z) and z <= self.z_cap

class ThickCappedZEllipsoid:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, thickness, z_ratio):
        """
        3D capped ellipsoid model.
        Dimensions are for inner capped ellipsoid.
        Long-axis is parallel to z-axis
        """
        inner = CappedZEllipsoid(x_min, x_max, y_min, y_max, z_min, z_max, z_ratio)
        outer = CappedZEllipsoid(x_min-thickness, x_max+thickness,
                                 y_min-thickness, y_max+thickness,
                                 z_min-thickness, z_max+thickness,
                                 z_ratio)
        self.inner = inner
        self.outer = outer
    
    def is_inside(self, x, y, z):
        return self.outer.is_inside(x, y, z) and not self.inner.is_inside(x, y, z)
    
