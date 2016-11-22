(* ::Package:: *)

BeginPackage["HexPlot`"]

fpath = $ScriptCommandLine[[2]]
prefix = $ScriptCommandLine[[3]]
shiftname = $ScriptCommandLine[[4]]
plotdata = Import[fpath, "RawJSON"]

a1 = {1/2, -Sqrt[3]/2}
a2 = {1/2, Sqrt[3]/2}
(* NOTE - assume that the d-grid has the same number of ds along a1 and a2 *)
N1 = Sqrt[Length[plotdata[["_ds"]]]]
N2 = N1

(* Assume origin is at A site if shift to B site is not specified *)
origin = If[shiftname == "B", {1/2, 1/(2 Sqrt[3])}, {0, 0}]

(* Return the Cartesian form of d + {-1, 0, 1}a1 + {-1, 0, 1}a2 *)
CartesianStar[d_] :=
	Module[{star = {}, mn},
		Do[AppendTo[star, (d[[1]] + mn[[1]])*a1 + (d[[2]] + mn[[2]])*a2],
			{mn, Tuples[{{-1, 0, 1}, {-1, 0, 1}}]}
		];
		star
	]

(* For each value in series, return {dx, dy, value} where d is the 
    corresponding element in _ds (and additional equivalent ds as given
    CartesianStar) *)
CartesianData[series_] :=
	Module[{xyv = {}, dval, d, value, v, vs},
		Do[d = dval[[1]]; value = dval[[2]];
			Do[vs = v + origin;
				AppendTo[xyv, {vs[[1]], vs[[2]], value}],
				{v, CartesianStar[d]}
			], {dval, Transpose[{plotdata[["_ds"]], series}]}
		];
		xyv
	]

(* Make and save the Cartesian cell plot for the data corresponding to label *)
HexPlot[label_] :=
	Module[{fixlabel = StringReplace[label, "/" -> "_"],
		xyv = CartesianData[plotdata[[label]]]},
		Export[prefix<>"_"<>fixlabel<>".png",
			Show[Graphics[{Thick, Arrow[{{0,0}, {1,0}}], Arrow[{{0,0}, {1/2,Sqrt[3]/2}}]}],
				ListDensityPlot[xyv, AspectRatio->Automatic, PlotLegends->Automatic,
					PlotRange->{{0,3/2},{0,Sqrt[3]/2},All},
					RegionFunction->Function[{x,y,z},x>=0&&x<=3/2&&y>=0&&y<=Sqrt[3]/2&&y<=Sqrt[3]*x&&y>=Sqrt[3]*x-Sqrt[3]],
					InterpolationOrder->0, Frame->None, ColorFunction->"DeepSeaColors",LabelStyle->{Bold,Black,18}]
				]
			]
	]

(* Ignore keys starting with "_" (these are special data such as the list of
    ds, not value series to plot) *)
HexPlotFilter[label_] := If[StringTake[label, 1] == "_", Null, HexPlot[label]]

plots = AssociationMap[HexPlotFilter, Keys[plotdata]]

EndPackage[]