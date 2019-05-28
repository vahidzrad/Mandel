
			lc = DefineNumber[ 0.01, Name "Parameters/lc" ];
			H = 1.;
			L = 1.;

			a=0.25;

		    Point(1) = {-L/2, H/2, 0, 5*lc};
		    Point(2) = {L/2, H/2, 0, 5*lc};
		    Point(3) = {L/2, -H/2, 0, 5*lc};
		    Point(4) = {-L/2, -H/2, 0, 5*lc};

		    Point(5) = {-a, 0, 0, 0.1*lc};
		    Point(6) = {a, 0, 0, 0.1*lc};


		    Line(1) = {1, 2};
		    Line(2) = {2, 3};
		    Line(3) = {3, 4};
		    Line(4) = {4, 1};
		    Line Loop(5) = {1, 2, 3, 4};

			Plane Surface(30) = {5};

			Line(6) = {5, 6};
		    Line{6} In Surface{30};

			Physical Surface(1) = {30};

			Physical Line(101) = {6};

