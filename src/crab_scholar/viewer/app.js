/* CrabScholar — Knowledge Graph Viewer App Logic */

var allNodes = nodes.get();
var allEdges = edges.get();
var hiddenEntityTypes = new Set();
var hiddenRelationTypes = new Set();
var minDegree = CRAB_CONFIG.defaultDegree;
var selectedNodeId = null;

// --- Stabilization: freeze physics after layout ---
network.once('stabilizationIterationsDone', function () {
    network.setOptions({ physics: { enabled: false } });
});

// --- Detail sidebar ---
var dp = document.getElementById('detail-panel');
var dt = document.getElementById('detail-title');
var db = document.getElementById('detail-body');

function esc(s) {
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}

function formatRelLabel(r) {
    return r.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, function (l) { return l.toUpperCase(); });
}

// --- Edge hover tooltip ---
network.on('hoverEdge', function (params) {
    var edge = edges.get(params.edge);
    if (!edge) return;
    var tip = document.getElementById('edge-tooltip');
    var label = (edge.source_name || edge.from) + '  →  ' + (edge.target_name || edge.to);
    var rt = formatRelLabel(edge.relation_type || 'UNKNOWN');
    tip.innerHTML = '<span style="color:var(--accent);font-weight:600">' + esc(rt) + '</span><br>' + esc(label);
    tip.style.display = 'block';
});
network.on('blurEdge', function () {
    document.getElementById('edge-tooltip').style.display = 'none';
});
document.addEventListener('mousemove', function (e) {
    var tip = document.getElementById('edge-tooltip');
    if (tip.style.display === 'block') {
        tip.style.left = (e.clientX + 16) + 'px';
        tip.style.top = (e.clientY - 10) + 'px';
    }
});

// --- Node click → detail with highlight ---
network.on('click', function (params) {
    if (params.nodes && params.nodes.length > 0) {
        var nid = params.nodes[0];
        selectedNodeId = nid;
        highlightConnected(nid);
        showNodeDetail(nid);
    } else if (params.edges && params.edges.length > 0) {
        showEdgeDetail(params.edges[0]);
    } else {
        // Clicked empty canvas
        clearHighlight();
        closeDetail();
    }
});

// --- Highlight connected nodes on selection ---
function highlightConnected(nodeId) {
    var connectedNodes = new Set([nodeId]);
    var connectedEdges = new Set();
    allEdges.forEach(function (e) {
        if (e.from === nodeId || e.to === nodeId) {
            connectedNodes.add(e.from);
            connectedNodes.add(e.to);
            connectedEdges.add(e.id);
        }
    });

    var nodeUpdates = allNodes.map(function (n) {
        if (connectedNodes.has(n.id)) {
            var isCenter = n.id === nodeId;
            return {
                id: n.id,
                opacity: 1.0,
                font: isCenter ? { size: 14, color: '#fff', strokeWidth: 3, strokeColor: 'rgba(0,0,0,0.7)' } : { size: 11, color: '#ccc', strokeWidth: 2, strokeColor: 'rgba(0,0,0,0.5)' },
                borderWidth: isCenter ? 3 : 2
            };
        } else {
            return { id: n.id, opacity: 0.08, font: { size: 0 }, borderWidth: 1 };
        }
    });
    nodes.update(nodeUpdates);

    var edgeUpdates = allEdges.map(function (e) {
        return { id: e.id, opacity: connectedEdges.has(e.id) ? 0.8 : 0.03 };
    });
    edges.update(edgeUpdates);
}

function clearHighlight() {
    selectedNodeId = null;
    var nodeUpdates = allNodes.map(function (n) {
        return { id: n.id, opacity: 1.0, font: { size: 0 }, borderWidth: 1.5 };
    });
    nodes.update(nodeUpdates);
    var edgeUpdates = allEdges.map(function (e) {
        return { id: e.id, opacity: undefined };
    });
    edges.update(edgeUpdates);
}

// --- Node detail panel ---
function showNodeDetail(nodeId) {
    var node = nodes.get(nodeId);
    if (!node) return;
    dt.textContent = node.full_name || node.label || nodeId;
    dp.classList.add('open');

    var connMap = {};
    allEdges.forEach(function (e) {
        var dir, nid, name;
        if (e.from === nodeId) { dir = '\u2192'; nid = e.to; name = e.target_name || e.to; }
        else if (e.to === nodeId) { dir = '\u2190'; nid = e.from; name = e.source_name || e.from; }
        else return;
        var key = dir + '|' + nid;
        if (!connMap[key]) connMap[key] = { rels: [], edges: [], name: name, dir: dir, nid: nid };
        if (connMap[key].rels.indexOf(e.relation_type) < 0) connMap[key].rels.push(e.relation_type);
        connMap[key].edges.push({
            relation_type: e.relation_type,
            confidence: e.edge_confidence || 0,
            evidence: e.full_evidence || ''
        });
    });
    var conns = [];
    for (var k in connMap) {
        var g = connMap[k];
        g.rels.sort();
        g.rel = g.rels.join(', ');
        conns.push(g);
    }
    conns.sort(function (a, b) {
        var aOut = a.dir === '\u2192' ? 0 : 1;
        var bOut = b.dir === '\u2192' ? 0 : 1;
        if (aOut !== bOut) return aOut - bOut;
        return a.rel.localeCompare(b.rel) || a.name.localeCompare(b.name);
    });

    var h = '';
    var tc = node.color || '#6366f1';
    if (typeof tc === 'object') tc = tc.background || '#6366f1';

    h += '<div class="d-field"><div class="d-label">Type</div>';
    h += '<span class="d-badge" style="background:' + esc(tc) + ';color:#080c18">' + esc(node.entity_type || 'UNKNOWN') + '</span></div>';

    var lines = (node.title || '').split('\n');
    for (var i = 1; i < lines.length; i++) {
        var ln = lines[i].trim();
        if (!ln) continue;
        var ci = ln.indexOf(': ');
        if (ci > 0) {
            h += '<div class="d-field"><div class="d-label">' + esc(ln.substring(0, ci)) + '</div>';
            h += '<div class="d-val">' + esc(ln.substring(ci + 2)) + '</div></div>';
        }
    }

    var outgoing = [], incoming = [];
    for (var j = 0; j < conns.length; j++) {
        if (conns[j].dir === '\u2190') incoming.push(conns[j]);
        else outgoing.push(conns[j]);
    }

    h += '<div class="d-field" style="margin-top:16px">';
    h += '<div class="d-label">Connections (' + conns.length + ')</div>';
    if (outgoing.length > 0) {
        h += '<div class="d-conn-group-header">\u2192 Outgoing (' + outgoing.length + ')</div>';
        for (var o = 0; o < outgoing.length; o++) h += buildConnRow(outgoing[o]);
    }
    if (incoming.length > 0) {
        h += '<div class="d-conn-group-header">\u2190 Incoming (' + incoming.length + ')</div>';
        for (var ii = 0; ii < incoming.length; ii++) h += buildConnRow(incoming[ii]);
    }
    h += '</div>';
    db.innerHTML = h;
}

function buildConnRow(c) {
    var relLabel = c.rels.map(function (r) { return formatRelLabel(r); }).join(', ');
    var h = '<div class="d-conn" data-nid="' + esc(c.nid) + '">';
    h += '<div class="d-conn-header" onclick="this.parentElement.classList.toggle(\'expanded\')">';
    h += '<div class="d-conn-summary">';
    h += '<span style="color:var(--text-muted);font-size:9px;letter-spacing:0.5px;font-weight:600">' + esc(relLabel).toUpperCase() + '</span><br>';
    h += '<span style="font-weight:500">' + esc(c.name) + '</span>';
    h += '</div>';
    h += '<button class="d-conn-nav" onclick="event.stopPropagation();focusNode(\'' + esc(c.nid).replace(/'/g, "\\'") + '\')" title="Go to ' + esc(c.name) + '">\u2192</button>';
    h += '</div>';
    h += '<div class="d-conn-detail">';
    for (var ei = 0; ei < c.edges.length; ei++) {
        var ed = c.edges[ei];
        h += '<div class="d-edge-card">';
        h += '<div class="d-label">' + esc(formatRelLabel(ed.relation_type)).toUpperCase() + '</div>';
        var parts = [];
        if (ed.confidence) parts.push(Math.round(ed.confidence * 100) + '% confidence');
        if (parts.length > 0) h += '<div class="d-val" style="color:var(--text-muted)">' + esc(parts.join(' · ')) + '</div>';
        if (ed.evidence) h += '<div class="d-evidence" style="margin-top:6px">' + esc(ed.evidence) + '</div>';
        h += '</div>';
    }
    h += '</div></div>';
    return h;
}

function showEdgeDetail(edgeId) {
    var edge = edges.get(edgeId);
    if (!edge) return;
    var rt = edge.relation_type || 'UNKNOWN';
    dt.textContent = formatRelLabel(rt);
    dp.classList.add('open');

    var ec = edge.color || '#6366f1';
    if (typeof ec === 'object') ec = ec.color || '#6366f1';

    var h = '';
    h += '<div class="d-field"><span class="d-badge" style="background:' + esc(ec) + ';color:#080c18">' + esc(rt) + '</span></div>';
    h += '<div class="d-field"><div class="d-label">From</div>';
    h += '<div class="d-val" style="cursor:pointer;color:var(--accent)" onclick="focusNode(\'' + esc(edge.from).replace(/'/g, "\\'") + '\')">' + esc(edge.source_name || edge.from) + '</div></div>';
    h += '<div class="d-field"><div class="d-label">To</div>';
    h += '<div class="d-val" style="cursor:pointer;color:var(--accent)" onclick="focusNode(\'' + esc(edge.to).replace(/'/g, "\\'") + '\')">' + esc(edge.target_name || edge.to) + '</div></div>';

    if (edge.edge_confidence) {
        h += '<div class="d-field"><div class="d-label">Confidence</div>';
        h += '<div class="d-val" style="font-family:JetBrains Mono,monospace;font-weight:600;color:var(--success)">' + Math.round(edge.edge_confidence * 100) + '%</div></div>';
    }
    if (edge.full_evidence) {
        h += '<div class="d-field"><div class="d-label">Evidence</div>';
        h += '<div class="d-evidence">' + esc(edge.full_evidence) + '</div></div>';
    }
    db.innerHTML = h;
}

function focusNode(nid) {
    network.focus(nid, { scale: 1.5, animation: { duration: 500, easingFunction: 'easeInOutCubic' } });
    network.selectNodes([nid]);
    selectedNodeId = nid;
    highlightConnected(nid);
    showNodeDetail(nid);
}

function closeDetail() {
    dp.classList.remove('open');
}

// --- Entity type toggle ---
function toggleEntityType(cb) {
    var type = cb.dataset.etype;
    if (cb.checked) hiddenEntityTypes.delete(type);
    else hiddenEntityTypes.add(type);
    applyFilters();
}

// --- Relation type toggle ---
function toggleRelationType(cb) {
    var type = cb.dataset.rtype;
    if (cb.checked) hiddenRelationTypes.delete(type);
    else hiddenRelationTypes.add(type);
    applyEdgeFilters();
}

// --- Entity color picker ---
function changeEntityColor(input) {
    var type = input.dataset.etypeColor;
    var color = input.value;
    var updates = [];
    allNodes.forEach(function (n) {
        if (n.entity_type === type) {
            updates.push({
                id: n.id,
                color: {
                    background: color,
                    border: 'rgba(255,255,255,0.1)',
                    highlight: { background: color, border: 'var(--accent)' }
                }
            });
        }
    });
    nodes.update(updates);
    allNodes = nodes.get();
}

function applyFilters() {
    var updates = [];
    allNodes.forEach(function (node) {
        var hidden = hiddenEntityTypes.has(node.entity_type) ||
            (node.node_degree || 0) < minDegree;
        updates.push({ id: node.id, hidden: hidden });
    });
    nodes.update(updates);
}

function applyEdgeFilters() {
    var updates = [];
    allEdges.forEach(function (edge) {
        var hidden = hiddenRelationTypes.has(edge.relation_type);
        updates.push({ id: edge.id, hidden: hidden });
    });
    edges.update(updates);
}

// --- Degree filter ---
function filterByDegree(val) {
    minDegree = parseInt(val, 10);
    document.getElementById('deg-val').textContent = minDegree;
    applyFilters();
}

// --- Search with glow effect ---
function searchEntity(query) {
    if (!query || query.length < 2) {
        clearHighlight();
        return;
    }
    query = query.toLowerCase();
    var matchIds = new Set();
    allNodes.forEach(function (n) {
        var name = (n.full_name || n.label || '').toLowerCase();
        if (name.includes(query)) matchIds.add(n.id);
    });
    var neighborIds = new Set();
    allEdges.forEach(function (e) {
        if (matchIds.has(e.from)) neighborIds.add(e.to);
        if (matchIds.has(e.to)) neighborIds.add(e.from);
    });
    var updates = allNodes.map(function (n) {
        if (matchIds.has(n.id)) {
            return {
                id: n.id, opacity: 1.0,
                font: { size: 14, color: '#fff', strokeWidth: 3, strokeColor: 'rgba(0,0,0,0.7)' },
                borderWidth: 3,
                shadow: { enabled: true, color: 'rgba(99,102,241,0.5)', size: 20 }
            };
        } else if (neighborIds.has(n.id)) {
            return { id: n.id, opacity: 0.5, font: { size: 10, color: '#aaa' }, borderWidth: 1.5, shadow: false };
        } else {
            return { id: n.id, opacity: 0.05, font: { size: 0 }, borderWidth: 1, shadow: false };
        }
    });
    nodes.update(updates);

    if (matchIds.size > 0) {
        var first = matchIds.values().next().value;
        network.focus(first, { scale: 1.2, animation: { duration: 500, easingFunction: 'easeInOutCubic' } });
    }
}

// --- Startup ---
(function () {
    applyFilters();
    applyEdgeFilters();
})();
