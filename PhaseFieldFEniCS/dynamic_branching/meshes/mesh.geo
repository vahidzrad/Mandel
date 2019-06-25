
            lc = DefineNumber[ 0.000125, Name "Parameters/lc" ];
            H = 40.0e-3;
            L = 100.0e-3;

            a = 50.0e-3;

            Point(1) = {0, 0, 0, 10*lc};
            Point(2) = {L, 0, 0, 10*lc};
            Point(3) = {L, H, 0, 10*lc};
            Point(4) = {0, H, 0, 10*lc};

            Point(5) = {0, H/2.0, 0, 1*lc};
            Point(6) = {L, H/2.0, 0, 1*lc};
            Point(7) = {L/2.0, H/2.0, 0, 1*lc};
            
            Line(1) = {1, 2};
            Line(2) = {2, 6};
            Line(3) = {6, 3};
            Line(4) = {3, 4};
            Line(5) = {4, 5};
            Line(6) = {5, 1};
            Line(7) = {5, 7};
            Line(8) = {7, 6};
            
            Line Loop(100) = {1, 2, -8, -7, 6};
            Line Loop(200) = {7, 8, 3, 4, 5};
            

            Plane Surface(1000) = {100};
            Plane Surface(2000) = {200};
            
            // Line(6) = {5, 6};
            
            // Line{6} In Surface{30};
            
            Physical Surface(1) = {1000};
            Physical Surface(2) = {2000};
            
            // Physical Line(101) = {6};

