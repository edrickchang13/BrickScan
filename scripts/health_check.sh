#!/bin/bash
# Check health of all BrickScan services
# Run from project root: ./scripts/health_check.sh

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Unicode characters (fallback to text if not supported)
PASS="✓"
FAIL="✗"
WARN="⚠"

# Check if unicode is supported
if ! locale charmap 2>/dev/null | grep -qi UTF-8; then
    PASS="[PASS]"
    FAIL="[FAIL]"
    WARN="[WARN]"
fi

echo "========================================="
echo "   BrickScan Health Check"
echo "========================================="
echo ""

# Initialize counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# Helper function to print status
print_status() {
    local status=$1
    local message=$2
    local color=$3

    if [ "$status" = "pass" ]; then
        echo -e "${GREEN}${PASS}${NC} ${color}${message}${NC}"
        ((PASS_COUNT++))
    elif [ "$status" = "fail" ]; then
        echo -e "${RED}${FAIL}${NC} ${color}${message}${NC}"
        ((FAIL_COUNT++))
    else
        echo -e "${YELLOW}${WARN}${NC} ${color}${message}${NC}"
        ((WARN_COUNT++))
    fi
}

# Check if running in docker-compose directory
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml not found in current directory"
    echo "Please run this script from the project root"
    exit 1
fi

# Check PostgreSQL
echo "Database Services:"
echo "-----------------"
if docker-compose exec -T db pg_isready -U postgres -d brickscan &>/dev/null 2>&1; then
    print_status "pass" "PostgreSQL: running"
else
    print_status "fail" "PostgreSQL: not responding or not running"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    print_status "pass" "Redis: running"
else
    print_status "fail" "Redis: not responding or not running"
fi

# Check Backend API
echo ""
echo "Backend Services:"
echo "-----------------"
if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    print_status "pass" "Backend API: running at http://localhost:8000"

    # Try to get health details
    HEALTH_JSON=$(curl -s http://localhost:8000/health 2>/dev/null)
    if [ -n "$HEALTH_JSON" ]; then
        DB_STATUS=$(echo "$HEALTH_JSON" | grep -o '"database":"[^"]*"' | cut -d'"' -f4)
        CACHE_STATUS=$(echo "$HEALTH_JSON" | grep -o '"cache":"[^"]*"' | cut -d'"' -f4)

        if [ -n "$DB_STATUS" ]; then
            if [ "$DB_STATUS" = "healthy" ] || [ "$DB_STATUS" = "ok" ]; then
                print_status "pass" "  └─ Database connection: OK"
            else
                print_status "fail" "  └─ Database connection: $DB_STATUS"
            fi
        fi

        if [ -n "$CACHE_STATUS" ]; then
            if [ "$CACHE_STATUS" = "healthy" ] || [ "$CACHE_STATUS" = "ok" ]; then
                print_status "pass" "  └─ Cache connection: OK"
            else
                print_status "fail" "  └─ Cache connection: $CACHE_STATUS"
            fi
        fi
    fi
else
    print_status "fail" "Backend API: not responding at http://localhost:8000"
fi

# Check DGX Vision Server (optional)
echo ""
echo "Optional Services:"
echo "------------------"
DGX_URL="${DGX_VISION_URL:-http://localhost:8001}"

if curl -sf "$DGX_URL/health" >/dev/null 2>&1; then
    print_status "pass" "DGX Vision Server: running at $DGX_URL"
else
    print_status "warn" "DGX Vision Server: not available at $DGX_URL (optional)"
fi

# Check Gemini fallback
if [ -n "$GOOGLE_API_KEY" ]; then
    print_status "pass" "Gemini API: credentials configured"
else
    print_status "warn" "Gemini API: not configured (DGX fallback only)"
fi

# Database tables
echo ""
echo "Database Tables:"
echo "----------------"
TABLES=$(docker-compose exec -T db psql -U postgres -d brickscan -c "
SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;
" 2>/dev/null | tail -n +3)

if [ -z "$TABLES" ]; then
    print_status "fail" "No tables found in database"
else
    TABLE_COUNT=$(echo "$TABLES" | wc -l)
    print_status "pass" "Found $TABLE_COUNT tables in database:"

    echo "$TABLES" | while read TABLE; do
        if [ -n "$TABLE" ]; then
            SIZE=$(docker-compose exec -T db psql -U postgres -d brickscan -c "
SELECT pg_size_pretty(pg_total_relation_size('$TABLE'::regclass));
" 2>/dev/null | tail -n 2 | head -n 1)

            if [ -n "$SIZE" ]; then
                echo "  ├─ $TABLE ($SIZE)"
            fi
        fi
    done
fi

# Check key tables
echo ""
echo "Key Tables Status:"
echo "------------------"
EXPECTED_TABLES=("users" "parts" "colors" "lego_sets" "inventory_items" "set_parts")

for table in "${EXPECTED_TABLES[@]}"; do
    COUNT=$(docker-compose exec -T db psql -U postgres -d brickscan -c "
SELECT COUNT(*) FROM $table;" 2>/dev/null | tail -n 2 | head -n 1 | tr -d ' ')

    if [ -z "$COUNT" ]; then
        print_status "fail" "$table: not accessible"
    elif [ "$COUNT" = "0" ]; then
        print_status "warn" "$table: empty (0 rows)"
    else
        print_status "pass" "$table: $COUNT rows"
    fi
done

# Docker images and versions
echo ""
echo "Docker Services:"
echo "----------------"
SERVICES=$(docker-compose ps --format "table {{.Service}}\t{{.Status}}" 2>/dev/null | tail -n +2)

if [ -z "$SERVICES" ]; then
    print_status "fail" "Docker Compose not responding"
else
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            SERVICE=$(echo "$line" | awk '{print $1}')
            STATUS=$(echo "$line" | awk '{print $2}')

            if echo "$STATUS" | grep -qi "up"; then
                print_status "pass" "$SERVICE: $STATUS"
            else
                print_status "fail" "$SERVICE: $STATUS"
            fi
        fi
    done <<< "$SERVICES"
fi

# Summary
echo ""
echo "========================================="
echo "Summary:"
echo "--------"
TOTAL=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))

echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo -e "${YELLOW}Warnings: $WARN_COUNT${NC}"
echo "Total: $TOTAL"

echo ""

# Exit with appropriate code
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Some services are not healthy. Please check logs.${NC}"
    exit 1
elif [ $WARN_COUNT -gt 0 ]; then
    echo -e "${YELLOW}All critical services running. Some optional services unavailable.${NC}"
    exit 0
else
    echo -e "${GREEN}All services healthy!${NC}"
    exit 0
fi
