import { Schema,Types,model } from "mongoose";

export interface User{
    name:string;
    email:string;
    profile:string;
    password:string;
    type:string;
    login_date:string;
    admin:boolean;
  }
  export const UserSchema=new Schema<User>(
    {
        name:{type:String,required:true},
        email:{type:String,required:true},
        profile:{type:String},
        password:{type:String},
        login_date:{type:String},
        type:{type:String,required:true},
        admin:{type:Boolean,default:false},
       
    },{
        toJSON:{
            virtuals:true
        },
        toObject:{
            virtuals:true
        },
        timestamps:true

    }
);
export const UserModel = model<User>('user', UserSchema);
