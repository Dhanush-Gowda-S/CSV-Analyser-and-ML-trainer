// ======================================================
// ADVANCED VISUALIZATION ENGINE (Auto Plugin Detection)
// ======================================================

(function () {

    /* -----------------------------
       LOAD DATA FROM HTML
    ------------------------------ */
    let raw = document.getElementById("chart-data").textContent;
    let dataset = [];
    try { dataset = JSON.parse(raw); }
    catch (e) { console.error("Invalid dataset JSON", e); }

    const colSummary = document.getElementById("colSummary");
    const vizNotes  = document.getElementById("vizNotes");
    const xcol      = document.getElementById("xcol");
    const ycol      = document.getElementById("ycol");
    const chartType = document.getElementById("chartType");
    const aggFunc   = document.getElementById("aggFunc");
    const topN      = document.getElementById("topN");
    const groupBy   = document.getElementById("groupBy");
    const canvas    = document.getElementById("vizCanvas");
    const genBtn    = document.getElementById("genBtn");
    const downloadBtn = document.getElementById("downloadBtn");

    let currentChart = null;

    if (!dataset || dataset.length === 0) {
        colSummary.innerHTML = "<p class='text-danger'>No data available.</p>";
        genBtn.disabled = true;
        return;
    }

    /* -----------------------------
       DETECT PLUGINS
    ------------------------------ */
    const SUPPORT = {
        boxplot:  !!Chart.controllers.boxplot || !!Chart.BoxPlotController,
        matrix:   !!Chart.controllers.matrix,
        histogram:!!Chart.controllers.histogram
    };

    console.log("PLUGIN SUPPORT:", SUPPORT);

    function updatePluginNotes() {
        let notes = [];

        if (!SUPPORT.boxplot) notes.push("❌ Boxplot plugin not available.");
        else notes.push("✔ Boxplot supported.");

        if (!SUPPORT.matrix) notes.push("❌ Heatmap plugin not available.");
        else notes.push("✔ Heatmap supported.");

        if (!SUPPORT.histogram) notes.push("❌ Histogram plugin not available.");
        else notes.push("✔ Histogram supported.");

        vizNotes.innerHTML = notes.map(n => `<li>${n}</li>`).join("");
    }

    updatePluginNotes();

    /* -----------------------------
       COLUMN TYPE DETECTION
    ------------------------------ */
    const sample = dataset.slice(0, Math.min(dataset.length, 300));
    const allCols = Object.keys(sample[0] || {});
    const colTypes = {};

    allCols.forEach(col => {
        let numericVotes = 0;
        sample.forEach(r => {
            let v = r[col];
            if (typeof v === "number") numericVotes++;
            else if (!isNaN(Number(v))) numericVotes++;
        });
        colTypes[col] = numericVotes / sample.length >= 0.6 ? "numeric" : "categorical";
    });

    /* -----------------------------
       POPULATE UI SELECTORS
    ------------------------------ */
    function fillSelectors() {
        allCols.forEach(c => {
            xcol.append(new Option(c, c));
            ycol.append(new Option(c, c));
            groupBy.append(new Option(c, c));
        });

        // Filter chart types if plugins missing
        const chartOptions = [
            ["bar","Bar"],
            ["grouped_bar","Grouped Bar"],
            ["line","Line"],
            ["scatter","Scatter"],
            ["pie","Pie"],
            ["doughnut","Doughnut"],
            ["histogram","Histogram"],
            ["boxplot","Boxplot"],
            ["heatmap","Heatmap"]
        ];

        chartOptions.forEach(([val,label]) => {
            if (val==="histogram" && !SUPPORT.histogram) return;
            if (val==="boxplot" && !SUPPORT.boxplot) return;
            if (val==="heatmap" && !SUPPORT.matrix) return;

            chartType.append(new Option(label,val));
        });
    }

    fillSelectors();

    /* -----------------------------
       SHOW COLUMN SUMMARY
    ------------------------------ */
    function renderColSummary() {
        const nums = allCols.filter(c => colTypes[c]==="numeric");
        const cats = allCols.filter(c => colTypes[c]==="categorical");

        colSummary.innerHTML = `
            <p><strong>Total columns:</strong> ${allCols.length}</p>
            <p><strong>Numeric:</strong> ${nums.join(", ")}</p>
            <p><strong>Categorical:</strong> ${cats.join(", ")}</p>
        `;
    }

    renderColSummary();

    /* -----------------------------
       UTILITY HELPERS
    ------------------------------ */
    function destroyChart() {
        if (currentChart) { currentChart.destroy(); currentChart = null; }
    }

    /* -----------------------------
       CHART GENERATION ENGINE
    ------------------------------ */
    function generateChart() {
        destroyChart();

        let type = chartType.value;
        let X = xcol.value;
        let Y = ycol.value;
        let agg = aggFunc.value;
        let N = Number(topN.value);
        let G = groupBy.value;

        let ctx = canvas.getContext("2d");

        if (!X) {
            alert("Select X-axis");
            return;
        }

        /* ----------------------------------
           HEATMAP
        ---------------------------------- */
        if (type === "heatmap") {
            if (!SUPPORT.matrix) {
                alert("Heatmap plugin not available.");
                return;
            }

            const numericCols = allCols.filter(c => colTypes[c]==="numeric");
            if (numericCols.length < 2) {
                alert("Need at least 2 numeric columns.");
                return;
            }

            // compute correlations
            function corr(a,b){
                let ax=[], bx=[];
                sample.forEach(r=>{
                    let av=Number(r[a]), bv=Number(r[b]);
                    if (!isNaN(av) && !isNaN(bv)){ ax.push(av); bx.push(bv); }
                });
                let n=ax.length;
                if(n<2) return 0;

                let ma=ax.reduce((s,v)=>s+v,0)/n;
                let mb=bx.reduce((s,v)=>s+v,0)/n;

                let num=0,da=0,db=0;
                for(let i=0;i<n;i++){
                    num+=(ax[i]-ma)*(bx[i]-mb);
                    da+=(ax[i]-ma)**2;
                    db+=(bx[i]-mb)**2;
                }
                return num/Math.sqrt(da*db);
            }

            let data=[];
            numericCols.forEach((r,i)=>{
                numericCols.forEach((c,j)=>{
                    data.push({x:j, y:i, v:corr(r,c)});
                });
            });

            currentChart = new Chart(ctx,{
                type:"matrix",
                data:{ datasets:[{ data }] },
                options:{
                    scales:{
                        x:{ type:"category", labels:numericCols },
                        y:{ type:"category", labels:numericCols }
                    }
                }
            });

            return;
        }

        /* ----------------------------------
           HISTOGRAM
        ---------------------------------- */
        if (type === "histogram") {
            if (!SUPPORT.histogram) {
                alert("Histogram plugin not available.");
                return;
            }

            if (!X || colTypes[X] !== "numeric") {
                alert("Histogram requires a numeric column.");
                return;
            }

            let values = dataset.map(r => Number(r[X])).filter(v => !isNaN(v));

            currentChart = new Chart(ctx, {
                type: "histogram",
                data: { datasets: [{ label: X, data: values }] },
                options:{ scales:{ y:{ beginAtZero:true } } }
            });

            return;
        }

        /* ----------------------------------
           PIE / DOUGHNUT
        ---------------------------------- */
        if (type==="pie" || type==="doughnut") {
            if (!X) { alert("Select category column"); return; }

            let counts = {};
            dataset.forEach(r=>{
                let key = r[X] ?? "__MISSING__";
                counts[key] = (counts[key]||0)+1;
            });

            let labels = Object.keys(counts);
            let values = labels.map(k=>counts[k]);

            currentChart = new Chart(ctx,{
                type:type,
                data:{ labels, datasets:[{ data:values }] }
            });

            return;
        }

        /* ----------------------------------
           SCATTER
        ---------------------------------- */
        if (type==="scatter") {
            if (!X || !Y) { alert("Select X and Y"); return; }
            if (colTypes[X] !== "numeric" || colTypes[Y] !== "numeric") {
                alert("Scatter requires numeric X and Y");
                return;
            }

            let points = dataset.map(r=>{
                let xv = Number(r[X]);
                let yv = Number(r[Y]);
                return (!isNaN(xv)&&!isNaN(yv)) ? {x:xv,y:yv} : null;
            }).filter(Boolean);

            currentChart = new Chart(ctx,{
                type:"scatter",
                data:{ datasets:[{ label:`${Y} vs ${X}`, data:points }] }
            });

            return;
        }

        /* ----------------------------------
           LINE
        ---------------------------------- */
        if (type==="line") {
            if (!X || !Y) { alert("Select X & Y"); return; }
            if (colTypes[Y] !== "numeric") { alert("Y must be numeric"); return; }

            let sorted = [...dataset];
            sorted.sort((a,b)=> (a[X] > b[X] ? 1 : -1));

            currentChart = new Chart(ctx,{
                type:"line",
                data:{
                    labels: sorted.map(r=>r[X]),
                    datasets:[{ label:`${Y} vs ${X}`, data:sorted.map(r=>r[Y]) }]
                }
            });

            return;
        }

        /* ----------------------------------
           BAR / GROUPED BAR
        ---------------------------------- */
        function aggregate(keyCol,valueCol){
            let map={};
            sample.forEach(r=>{
                let k=r[keyCol] ?? "__MISSING__";
                let v=Number(r[valueCol]);
                if (!map[k]) map[k]=[];
                if (!isNaN(v)) map[k].push(v);
            });

            let rows=Object.entries(map).map(([k,arr])=>{
                if (agg==="count") return [k,arr.length];
                if (agg==="sum")   return [k,arr.reduce((a,b)=>a+b,0)];
                if (agg==="mean")  return [k,arr.reduce((a,b)=>a+b,0)/arr.length];
                if (agg==="min")   return [k,Math.min(...arr)];
                if (agg==="max")   return [k,Math.max(...arr)];
            });

            rows.sort((a,b)=>b[1]-a[1]);
            if (N>0) rows=rows.slice(0,N);

            return rows;
        }

        if (type==="bar") {
            if (!X || !Y) { alert("Select X & Y"); return; }

            let rows = aggregate(X,Y);
            let labels = rows.map(r=>r[0]);
            let values = rows.map(r=>r[1]);

            currentChart = new Chart(ctx,{
                type:"bar",
                data:{ labels, datasets:[{ label:`${agg} of ${Y}`, data:values }] }
            });

            return;
        }

        /* ----------------------------------
           BOXPLOT
        ---------------------------------- */
        if (type==="boxplot") {
            if (!SUPPORT.boxplot){
                alert("Boxplot plugin not available.");
                return;
            }

            if (!X || !Y) { alert("Select X & Y"); return; }
            if (colTypes[Y] !== "numeric"){
                alert("Y must be numeric for boxplot");
                return;
            }

            let groups = {};
            sample.forEach(r=>{
                let key = r[X] ?? "__MISSING__";
                let val = Number(r[Y]);
                if (!isNaN(val)) {
                    if (!groups[key]) groups[key]=[];
                    groups[key].push(val);
                }
            });

            let labels = Object.keys(groups);
            let data = labels.map(l=>{
                let arr=groups[l].sort((a,b)=>a-b);
                return {
                    min: arr[0],
                    q1:  quantile(arr,0.25),
                    median: quantile(arr,0.50),
                    q3: quantile(arr,0.75),
                    max: arr[arr.length-1]
                };
            });

            function quantile(arr,q){
                let pos=(arr.length-1)*q;
                let base=Math.floor(pos);
                let rest=pos-base;
                return arr[base+1]!==undefined ?
                    arr[base]+rest*(arr[base+1]-arr[base]) :
                    arr[base];
            }

            currentChart = new Chart(ctx,{
                type:"boxplot",
                data:{ labels, datasets:[{ label:`${Y} by ${X}`, data }] }
            });

            return;
        }

        alert("Chart type not implemented.");
    }

    /* -----------------------------
       DOWNLOAD PNG
    ------------------------------ */
    downloadBtn.addEventListener("click",()=>{
        if (!currentChart) return alert("Generate a chart first.");
        const link=document.createElement("a");
        link.href=currentChart.toBase64Image();
        link.download="chart.png";
        link.click();
    });

    /* -----------------------------
       GENERATE BUTTON
    ------------------------------ */
    genBtn.addEventListener("click",(ev)=>{
        ev.preventDefault();
        generateChart();
    });

})();
