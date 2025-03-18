import psutil

def scan_processes():
    suspicious_processes = []
    # Scan all running processes
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            info = proc.info
            # Define keywords that are considered suspicious
            keywords = ['tor', 'proxy', 'vpn']
            name = (info.get('name') or '').lower()
            cmdline = ' '.join(info.get('cmdline') or []).lower()
            if any(keyword in name for keyword in keywords) or any(keyword in cmdline for keyword in keywords):
                suspicious_processes.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return suspicious_processes

def scan_network():
    suspicious_connections = []
    # Get network connections (inet: IPv4 and IPv6)
    connections = psutil.net_connections(kind='inet')
    for conn in connections:
        if conn.raddr:
            # Example: Flag if remote port is Tor's default (9050)
            if conn.raddr.port == 9050:
                suspicious_connections.append({
                    'pid': conn.pid,
                    'raddr': f"{conn.raddr.ip}:{conn.raddr.port}",
                    'status': conn.status
                })
    return suspicious_connections

def run_detections():
    processes = scan_processes()
    networks = scan_network()
    findings = {
        'suspicious_processes': processes,
        'suspicious_connections': networks
    }
    return findings
