using PGFPlots
using Interact
using JLD2
using Colors
using ColorBrewer
using SparseArrays

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
        ymin = -600.0,
        ymax = 600.0,
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
            PGFPlots.save(filename, g2, include_preamble=:false)
        end

        return g
    end
end

function get_belief(state, grid)
	belief = spzeros(nS, 1)
    indices, weights = interpolants(grid, state)
    for i = 1:length(indices)
        belief[indices[i]] = weights[i]
    end
    return belief
end