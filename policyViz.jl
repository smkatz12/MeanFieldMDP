resetPGFPlotsPreamble()
pushPGFPlotsPreamble("\\usepackage{aircraftshapes}")

"""
----------------------------------
Vertical
----------------------------------
"""

function viz_vertical_policy(Q)
	vizgrid = RectangleGrid(hs, ḣ₀s, ḣ₁s, vτs)

	ra_1 = RGB(1.,1.,1.) # white
	ra_2 = RGB(1.0/255.0,88.0/255.0,202.0/255.0) # dark blue
    ra_3 = RGB(190.0/255.0,0.0/255.0,0.0/255.0) # dark red

    colors = [ra_1, ra_2, ra_3]
    bg_colors = [RGB(1.0, 1.0, 1.0)]

    # Create scatter plot classes for color key
    sc_string = "{"
    for i=1:length(colors)
        define_color("ra_$i",  colors[i])
        if i==1
            sc_string *= "ra_$i={mark=square, style={black, mark options={fill=ra_$i}, mark size=6}},"
        else
            sc_string *= "ra_$i={style={ra_$i, mark size=6}},"
        end
    end

    # Color key as a scatter plot
    sc_string=sc_string[1:end-1]*"}"
    xx = [-1.5, -1.5, -1.5]
    yy = [1.65, 1.15, 0.65]
    zz = ["ra_1","ra_2","ra_3"]
    sc = string(sc_string)

    currSavePlot = 0

    @manipulate for fileName in textbox(value="myFile.pdf",label="File Name") |> onchange,
		savePlot in button("Save Plot"), 
    	nbin = 100,
        xmin = 0.0,
        xmax = 40.0,
        ymin = -2000.0,
        ymax = 2000.0,
        ḣ₀ = collect(-100:10:100),
        ḣ₁ = collect(-100:10:100)

        function get_heat(x, y)
        	bel = get_belief([y,ḣ₀,ḣ₁,x], vizgrid)
        	qvals = zeros(length(vactions))
        	for i = 1:length(bel.rowval)
                qvals += bel[bel.rowval[i]]*Q[bel.rowval[i],:]
            end
            #println(qvals)
            return vactions[argmax(qvals)]
        end

        g = GroupPlot(2, 1, groupStyle = "horizontal sep=3cm")
        push!(g, Axis([
            Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), zmin = 1, zmax = 3,
            xbins = nbin, ybins = nbin, colormap = ColorMaps.RGBArrayMap(colors), colorbar=false),
            ], xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax, width="10cm", height="8cm", 
               xlabel="Tau (s)", ylabel="Relative Alt (ft)", title="Table Advisories"))

        # Create Color Key
            f = (x,y)->x # Dummy function for background white image
            push!(g, Axis([
                Plots.Image(f, (-2,2), (-2,2),colormap = ColorMaps.RGBArrayMap(bg_colors),colorbar=false),
                Plots.Scatter(xx, yy, zz, scatterClasses=sc),
                Plots.Node("RA 1: COC ",0.15,0.915,style="black,anchor=west", axis="axis description cs"),
                Plots.Node("RA 2: CL1500 ",0.15,0.790,style="black,anchor=west", axis="axis description cs"),
                Plots.Node("RA 3: DES1500",0.15,0.665,style="black,anchor=west", axis="axis description cs")
                ],width="10cm",height="8cm", hideAxis =true, title="KEY"))

        if savePlot > currSavePlot
        	g2 = GroupPlot(2, 1, groupStyle = "horizontal sep=3cm")
        	push!(g2, Axis([
	            Plots.Image(get_heat, (xmin, xmax), (ymin, ymax), zmin = 1, zmax = 3,
	            xbins = nbin, ybins = nbin, colormap = ColorMaps.RGBArrayMap(colors), colorbar=false),
	            ], xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax, width="10cm", height="8cm", 
	               xlabel="Tau (s)", ylabel="Relative Alt (ft)", title="Table Advisories"))

        # Create Color Key
            f = (x,y)->x # Dummy function for background white image
            push!(g2, Axis([
                Plots.Image(f, (-2,2), (-2,2),colormap = ColorMaps.RGBArrayMap(bg_colors),colorbar=false),
                Plots.Scatter(xx, yy, zz, scatterClasses=sc),
                Plots.Node("RA 1: COC ",0.15,0.915,style="black,anchor=west", axis="axis description cs"),
                Plots.Node("RA 2: CL1500 ",0.15,0.790,style="black,anchor=west", axis="axis description cs"),
                Plots.Node("RA 3: DES1500",0.15,0.665,style="black,anchor=west", axis="axis description cs")
                ],width="10cm",height="8cm", hideAxis =true, title="KEY"))
            PGFPlots.save(fileName, g2, include_preamble=:false)
        end

        return g
    end
end

"""
----------------------------------
Horizontal
----------------------------------
"""

function viz_horizontal_policy(Q)
	vizgrid = RectangleGrid(rs, θs, ψs, hτs)

	ra_1 = RGB(1.,1.,1.) # white
	ra_2 = RGB(255.0/255.0,128.0/255.0,0.0/255.0) # orange
    ra_3 = RGB(0.0/255.0,153.0/255.0,0.0/255.0) # green

    ra_colors = [ra_1, ra_2, ra_3]
    bg_colors = [RGB(1.0, 1.0, 1.0)]

    # Create scatter plot classes for color key
    sc_string = "{"
    for i=1:length(ra_colors)
        define_color("ra_$i",  ra_colors[i])
        if i==1
            sc_string *= "ra_$i={mark=square, style={black, mark options={fill=ra_$i}, mark size=6}},"
        else
            sc_string *= "ra_$i={style={ra_$i, mark size=6}},"
        end
    end

    # Color key as a scatter plot
    sc_string=sc_string[1:end-1]*"}"
    xx = [-1.5, -1.5, -1.5]
    yy = [1.65, 1.15, 0.65]
    zz = ["ra_1","ra_2","ra_3"]
    sc = string(sc_string)

    currSavePlot = 0

    @manipulate for fileName in textbox(value="myFile.pdf",label="File Name") |> onchange,
		savePlot in button("Save Plot"), 
    	nbin = 100,
    	zoom = 5.0,
        xshift = 0.0,
        yshift = 0.0,
        xscale = 1.0,
        yscale = 1.0,
        ψ = ψs,
        τ = hτs

        # Ensure that zoom and scale factors don't result in dividing by zero or negative numbers
        zoom = zoom <0.1 ? 0.1 : zoom
        xscale = xscale <0.1 ? 0.1 : xscale
        yscale = yscale <0.1 ? 0.1 : yscale

        RANGEMAX = maximum(rs)

        function get_heat(x, y)
        	r = √(x^2 + y^2)
        	θ = atan(y, x)
        	bel = get_belief([r, θ, ψ, τ], vizgrid)
        	qvals = zeros(length(hactions))
        	for i = 1:length(bel.rowval)
                qvals += bel[bel.rowval[i]]*Q[bel.rowval[i],:]
            end
            #println(qvals)
            return hactions[argmax(qvals)]
        end

		g = GroupPlot(2, 1, style="height={8cm}, width={8cm}",groupStyle = "horizontal sep=2cm")
        push!(g, Axis([
                        Plots.Image(get_heat, (-1*RANGEMAX/zoom/xscale - xshift, RANGEMAX/zoom/xscale- xshift), 
                            (-1*RANGEMAX/zoom/yscale- yshift, RANGEMAX/zoom/yscale- yshift), 
                            zmin = 1, zmax = 3,
                            xbins = nbin, ybins = nbin,
                            colormap = ColorMaps.RGBArrayMap(ra_colors), colorbar=false),
                        Plots.Command(getACString(0.0,0.0,0.0,"black","white")),
                        Plots.Command(getACString(rad2deg(ψ),RANGEMAX*0.82/zoom/xscale - xshift,RANGEMAX*0.82/zoom/yscale- yshift,"red","black"))
                        ], xlabel="East (ft)", ylabel="North (ft)", title="Policy"))

        # Create Color Key
        f = (x,y)->x # Dummy function for background white image
        push!(g, Axis([
            Plots.Image(f, (-2,2), (-2,2),colormap = ColorMaps.RGBArrayMap(bg_colors),colorbar=false),
            Plots.Scatter(xx, yy, zz, scatterClasses=sc),
            Plots.Node("RA 1: COC ",0.15,0.915,style="black,anchor=west", axis="axis description cs"),
            Plots.Node("RA 2: SL ",0.15,0.790,style="black,anchor=west", axis="axis description cs"),
            Plots.Node("RA 3: SR",0.15,0.665,style="black,anchor=west", axis="axis description cs")
            ],width="10cm",height="8cm", hideAxis =true, title="KEY"))

        if savePlot > currSavePlot
        	g2 = GroupPlot(2, 1, style="height={8cm}, width={8cm}",groupStyle = "horizontal sep=2cm")

            push!(g2, Axis([
                            Plots.Image(get_heat, (-1*RANGEMAX/zoom/xscale - xshift, RANGEMAX/zoom/xscale- xshift), 
                                (-1*RANGEMAX/zoom/yscale- yshift, RANGEMAX/zoom/yscale- yshift), 
                                zmin = 1, zmax = 3,
                                xbins = nbin, ybins = nbin,
                                colormap = ColorMaps.RGBArrayMap(ra_colors), colorbar=false),
                            Plots.Command(getACString(0.0,0.0,0.0,"black","white")),
                            Plots.Command(getACString(rad2deg(ψ),RANGEMAX*0.82/zoom/xscale - xshift,RANGEMAX*0.82/zoom/yscale- yshift,"red","black"))
                            ], xlabel="East (ft)", ylabel="North (ft)", title="Policy"))

            # Create Color Key
            f = (x,y)->x # Dummy function for background white image
            push!(g2, Axis([
                Plots.Image(f, (-2,2), (-2,2),colormap = ColorMaps.RGBArrayMap(bg_colors),colorbar=false),
                Plots.Scatter(xx, yy, zz, scatterClasses=sc),
                Plots.Node("RA 1: COC ",0.15,0.915,style="black,anchor=west", axis="axis description cs"),
                Plots.Node("RA 2: SL ",0.15,0.790,style="black,anchor=west", axis="axis description cs"),
                Plots.Node("RA 3: SR",0.15,0.665,style="black,anchor=west", axis="axis description cs")
                ],width="10cm",height="8cm", hideAxis =true, title="KEY"))

            PGFPlots.save(fileName, g2, include_preamble=:false)
        end

        return g
    end
end

"""
----------------------------------
Support functions
----------------------------------
"""

function get_belief(state, grid)
	belief = spzeros(length(grid), 1)
    indices, weights = interpolants(grid, state)
    for i = 1:length(indices)
        belief[indices[i]] = weights[i]
    end
    return belief
end

function getACString(theta,x,y,fill,draw,width=2.0)
    return "\\node[aircraft top,fill="*fill*",draw="*draw*", minimum width="*string(width)*"cm,rotate="*string(theta)*",scale = 0.35] at (axis cs:"*string(x)*", "*string(y)*") {};"
end