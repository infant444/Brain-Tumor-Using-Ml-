import { Schema, Types, model } from "mongoose";


export interface chatroom{
    name:string;
    send:string;
    responce:string;
    time:Date;
    userid:Types.ObjectId;
}
  
export const ChatroomSchema=new Schema<chatroom>({
    name:{type:String,required:true},
    send:{type:String,required:true},
    responce:{type:String,required:true},
    time:{type:Date,required:true},
    userid:{type:Schema.Types.ObjectId,required:true},
},{
    toJSON:{
        virtuals:true
    },
    toObject:{
        virtuals:true
    },
    timestamps:true
});

export const ChatRoomModel = model<chatroom>('ChatRoom', ChatroomSchema);
