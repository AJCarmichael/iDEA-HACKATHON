import google.generativeai as genai

def get_reasoning(findings):
    # Build detailed sections for processes and network connections.
    processes_section = ""
    if findings.get('suspicious_processes'):
        processes_section = "\n".join([f"PID {proc['pid']}: {proc['name']}" for proc in findings['suspicious_processes']])
    else:
        processes_section = "None detected."

    network_section = ""
    if findings.get('suspicious_connections'):
        network_section = "\n".join([f"PID {conn['pid']}: {conn['raddr']} ({conn['status']})" for conn in findings['suspicious_connections']])
    else:
        network_section = "None detected."

    # Construct a long and specific prompt with detailed instructions.
    prompt = f"""
You are an advanced security threat analysis system specializing in detecting insider threats within enterprise environments. You have been provided with detailed system monitoring data consisting of running processes and network connection details. Your task is to perform an in-depth analysis based on the following guidelines:

1. Analyze each suspicious process and network connection for indicators of insider threat activities. Consider the following aspects:
   - For processes: Look for usage of obfuscation tools such as dark web routing (e.g., Tor), proxies, VPNs, or any unusual command line parameters.
   - For network connections: Evaluate connection endpoints, remote IP addresses, port usage (e.g., port 9050 for Tor), and connection statuses.
2. Determine the potential intent behind these activities. Ask yourself:
   - Could these processes or connections be used to exfiltrate data or bypass network security?
   - Are they consistent with normal user behavior or do they deviate significantly from typical patterns?
3. Provide a comprehensive risk assessment report including:
   - A detailed explanation for each suspicious activity.
   - An overall threat level score on a scale from 0 (no threat) to 10 (imminent threat).
   - Specific recommendations for further investigation or immediate action to mitigate risk.
   - Any additional insights that could help a cybersecurity team understand the potential insider threat.

Data Provided:
---------------------------
Suspicious Processes:
{processes_section}

Suspicious Network Connections:
{network_section}
---------------------------

Please generate a detailed and well-structured report strictly in plain text, ensuring that each recommendation is actionable and the reasoning is clear and precise.
"""

    try:
        # Configure the Gemini API key (update with your actual API key)
        GEMINI_API_KEY = "AIzaSyDVJdRye4ECAFhpd2Lib7rnv-B-tRl5BPw"
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Use the Gemini model with a specific variant (modify if needed per your documentation)
        response = genai.GenerativeModel("gemini-2.0-flash-thinking-exp").generate_content(prompt)
        
        if response and response.text:
            return response.text
        else:
            return "No reasoning provided."
    except Exception as e:
        return f"Error calling Gemini API: {e}"
