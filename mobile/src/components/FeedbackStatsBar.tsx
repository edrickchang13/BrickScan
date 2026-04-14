/**
 * FeedbackStatsBar — small "X corrections contributed" counter.
 *
 * HOW TO ADD TO ScanHistoryScreen.tsx:
 * ------------------------------------
 * 1. Import at the top:
 *      import { FeedbackStatsBar } from '@/components/FeedbackStatsBar';
 *
 * 2. Add state for the counter:
 *      const [corrections, setCorrections] = useState<number | null>(null);
 *
 * 3. Inside useFocusEffect (after loadRecentScans), add:
 *      getFeedbackStats()
 *        .then(s => setCorrections(s.totalCorrections))
 *        .catch(() => {});
 *
 * 4. Add import for getFeedbackStats:
 *      import { getFeedbackStats } from '@/services/feedbackApi';
 *
 * 5. Render the bar just BELOW the header (before the FlatList):
 *      <FeedbackStatsBar corrections={corrections} />
 *
 * That's it — no other changes needed.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface FeedbackStatsBarProps {
  corrections: number | null;
}

export const FeedbackStatsBar: React.FC<FeedbackStatsBarProps> = ({ corrections }) => {
  if (corrections === null) return null;   // still loading

  return (
    <View style={styles.bar}>
      <Ionicons name="school-outline" size={13} color="#6366F1" />
      <Text style={styles.text}>
        {corrections === 0
          ? 'No corrections yet — tap "No, fix it" after a scan to help improve BrickScan'
          : `${corrections} correction${corrections === 1 ? '' : 's'} contributed to training`}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    marginHorizontal: 16,
    marginBottom: 8,
    backgroundColor: '#EEF2FF',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  text: {
    flex: 1,
    fontSize: 12,
    fontWeight: '600',
    color: '#4F46E5',
    lineHeight: 16,
  },
});
