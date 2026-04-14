import { create } from 'zustand';
import { apiClient } from '@/services/api';
import { InventoryItem } from '@/types';

interface InventoryStoreState {
  items: InventoryItem[];
  isLoading: boolean;
  error: string | null;
  fetchInventory: () => Promise<void>;
  addItem: (partNum: string, colorId: string, quantity: number, colorName?: string, colorHex?: string) => Promise<InventoryItem>;
  updateItem: (id: string, quantity: number) => Promise<InventoryItem>;
  removeItem: (id: string) => Promise<void>;
  clearInventory: () => void;
}

export const useInventoryStore = create<InventoryStoreState>((set, get) => ({
  items: [],
  isLoading: false,
  error: null,

  fetchInventory: async () => {
    set({ isLoading: true, error: null });
    try {
      const items = await apiClient.getInventory();
      set({ items, isLoading: false });
    } catch (error: any) {
      const errorMessage = error?.message || 'Failed to fetch inventory';
      set({ error: errorMessage, isLoading: false });
      throw error;
    }
  },

  addItem: async (partNum: string, colorId: string, quantity: number, colorName?: string, colorHex?: string) => {
    const optimisticItem: InventoryItem = {
      id: `temp-${Date.now()}`,
      partNum,
      partName: '',
      colorId,
      colorName: colorName || '',
      colorHex: colorHex || '#000000',
      quantity,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    set((state) => ({
      items: [...state.items, optimisticItem],
    }));

    try {
      const newItem = await apiClient.addToInventory(partNum, colorId, quantity, colorName, colorHex);
      set((state) => ({
        items: state.items.map((item) =>
          item.id === optimisticItem.id ? newItem : item
        ),
      }));
      return newItem;
    } catch (error: any) {
      set((state) => ({
        items: state.items.filter((item) => item.id !== optimisticItem.id),
      }));
      throw error;
    }
  },

  updateItem: async (id: string, quantity: number) => {
    const oldItems = get().items;
    const itemIndex = oldItems.findIndex((item) => item.id === id);

    if (itemIndex >= 0) {
      const oldQuantity = oldItems[itemIndex].quantity;
      set((state) => ({
        items: state.items.map((item) =>
          item.id === id ? { ...item, quantity } : item
        ),
      }));

      try {
        const updatedItem = await apiClient.updateInventory(id, quantity);
        set((state) => ({
          items: state.items.map((item) =>
            item.id === id ? updatedItem : item
          ),
        }));
        return updatedItem;
      } catch (error: any) {
        set((state) => ({
          items: state.items.map((item) =>
            item.id === id ? { ...item, quantity: oldQuantity } : item
          ),
        }));
        throw error;
      }
    }
    throw new Error('Item not found');
  },

  removeItem: async (id: string) => {
    const oldItems = get().items;
    set((state) => ({
      items: state.items.filter((item) => item.id !== id),
    }));

    try {
      await apiClient.deleteFromInventory(id);
    } catch (error: any) {
      set({ items: oldItems });
      throw error;
    }
  },

  clearInventory: () => {
    set({ items: [], error: null });
  },
}));
