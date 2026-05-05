from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.services.anomaly_service import (
    run_pipeline, get_latest_anomalies, get_anomaly_explanation, 
    get_metrics, reset_state, GLOBAL_STATE
)
import os

router = APIRouter()

def auto_map_columns(df):
    """
    Heuristically maps uploaded CSV columns to exactly match the required schema:
    'timestamp', 'user', 'event_type', 'details'
    """
    col_mapping = {}
    
    aliases = {
        'timestamp': ['time', 'date', 'datetime', 'logtime', 'created_at', '_time', 'ts', 'timestamp'],
        'user': ['username', 'employee', 'employee_id', 'userid', 'account', 'actor', 'src_user', 'user'],
        'event_type': ['event', 'action', 'activity', 'type', 'category', 'operation', 'event_type', 'eventtype'],
        'details': ['description', 'message', 'msg', 'info', 'note', 'content', 'data', 'details', 'file_tree', 'filename', 'to', 'pc']
    }
    
    available_cols = list(df.columns)
    
    for target, target_aliases in aliases.items():
        if target in available_cols:
            continue
            
        match_found = False
        for col in available_cols:
            clean_col = str(col).lower().replace(' ', '').replace('_', '')
            for alias in target_aliases:
                clean_alias = alias.replace('_', '')
                if clean_col == clean_alias or clean_alias in clean_col:
                    col_mapping[col] = target
                    available_cols.remove(col)
                    match_found = True
                    break
            if match_found:
                break
                
    if col_mapping:
        df = df.rename(columns=col_mapping)
        print(f"Auto-mapped columns using aliases: {col_mapping}")
        
    # Fully dynamic fallback: If we still don't have the required columns,
    # generate synthetic equivalents so ANY file template analyzes successfully
    required_cols = {'timestamp', 'user', 'event_type', 'details'}
    missing = required_cols - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing required schema columns after auto-mapping: {missing}. Please upload a CSV with the required fields: timestamp, user, event_type, details.")

    return df


@router.post("/reset")
async def reset_pipeline():
    """
    Resets the ML pipeline and clears all data.
    """
    try:
        reset_state()
        return {"status": "success", "message": "System state reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload_csv")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Accepts a custom CSV file and runs it through the ML pipeline.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        import pandas as pd
        # The user requested to allow huge files, so we expand the cap to 100,000 rows.
        # This will take ~2-3 minutes to run the IsolationForest and Autoencoder on standard machines.
        df = pd.read_csv(file.file, nrows=100000)
        df.columns = df.columns.astype(str).str.strip().str.lower()
        
        # Check if the file had data URI prefix on the first column header due to Safari issue
        if len(df.columns) > 0 and df.columns[0].startswith("data:"):
            df.columns.values[0] = df.columns[0].split(',')[-1]
            
        # Auto-map columns using heuristics
        df = auto_map_columns(df)
        
        # basic validation
        required_cols = {'timestamp', 'user', 'event_type', 'details'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
            
        # run pipeline with custom df
        try:
            run_pipeline(custom_df=df)
        except ValueError as inner_err:
            raise HTTPException(status_code=400, detail=str(inner_err))
        except Exception as inner_err:
            import traceback
            print(f"[PIPELINE ERROR] {str(inner_err)}")
            print(traceback.format_exc())
            raise
        metrics = get_metrics()
        return {"status": "success", "message": "Custom data analyzed successfully", "metrics": metrics}
    except Exception as e:
        import traceback
        print(f"[UPLOAD ERROR] {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomalies")
async def fetch_anomalies():
    """
    Returns the most recent anomalies detected.
    """
    anomalies = get_latest_anomalies(top_n=50)
    return {"anomalies": anomalies}

@router.get("/metrics")
async def fetch_metrics():
    """
    Returns system metrics.
    """
    return get_metrics()

@router.get("/explain/{log_id}")
async def explain_anomaly(log_id: int):
    """
    Returns SHAP feature importance for a specific anomaly.
    """
    explanation = get_anomaly_explanation(log_id)
    if "error" in explanation:
        raise HTTPException(status_code=400, detail=explanation["error"])
    return {"explanation": explanation}

@router.get("/graph")
async def get_graph_html(filter: str = "all"):
    """
    Returns the interactive PyVis HTML file dynamically based on the filter.
    """
    df = GLOBAL_STATE.get('raw_df')
    if df is None:
        raise HTTPException(status_code=404, detail="Graph not generated yet. Run simulation first.")
        
    from app.services.graph_service import build_behavioral_graph, export_graph_to_pyvis
    
    # Build graph with the selected filter
    graph = build_behavioral_graph(df, filter_type=filter)
    
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    filename = f"graph_{filter}.html"
    graph_path = export_graph_to_pyvis(graph, output_dir=static_dir, filename=filename)
    
    return FileResponse(graph_path)